import inspect

import dask.array as da
import numpy as np
import xarray as xr
from scipy import ndimage

from multiview_stitcher import msi_utils
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher.misc_utils import clear_cupy_memory

try:
    import cupy as cp
    import cupyx.scipy.ndimage
except ImportError:
    cp = None


def _normalize_pixel_value(value, ndim, name):
    if np.isscalar(value):
        return tuple(float(value) for _ in range(ndim))

    value = tuple(value)
    if len(value) != ndim:
        raise ValueError(f"{name} must have {ndim} values, got {len(value)}.")

    return tuple(float(v) for v in value)


def _compute_point_indices(mask):
    if isinstance(mask, da.Array):
        arr = da.argwhere(mask).compute()
        if cp is not None and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return np.asarray(arr)

    if cp is not None and isinstance(mask, cp.ndarray):
        return np.argwhere(cp.asnumpy(mask))
    return np.argwhere(np.asarray(mask))


def _function_accepts_kwarg(func, kwarg):
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False

    if kwarg in signature.parameters:
        return True

    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )


def _prepare_detection_func_kwargs(
    detection_func,
    detection_func_kwargs,
    threshold_rel,
    threshold_abs,
    cupy,
):
    if detection_func_kwargs is None:
        detection_func_kwargs = {}
    else:
        detection_func_kwargs = dict(detection_func_kwargs)

    optional_kwargs = {
        "threshold_rel": threshold_rel,
        "threshold_abs": threshold_abs,
        "cupy": cupy,
    }
    for key, value in optional_kwargs.items():
        if (
            key not in detection_func_kwargs
            and _function_accepts_kwarg(detection_func, key)
        ):
            detection_func_kwargs[key] = value

    return detection_func_kwargs


def _normalize_overlap(overlap, sdims):
    if isinstance(overlap, dict):
        return tuple(int(np.ceil(overlap[dim])) for dim in sdims)

    if np.isscalar(overlap):
        return tuple(int(np.ceil(overlap)) for _ in sdims)

    overlap = tuple(overlap)
    if len(overlap) != len(sdims):
        raise ValueError(
            f"detection_overlap must have {len(sdims)} values, "
            f"got {len(overlap)}."
        )

    return tuple(int(np.ceil(value)) for value in overlap)


def _get_detection_overlap(
    detection_func,
    target_size_pixels,
    detection_func_kwargs,
    detection_overlap,
    sdims,
):
    has_overlap = hasattr(detection_func, "required_overlap")
    if detection_overlap is None and has_overlap:
        detection_overlap = detection_func.required_overlap(
            target_size_pixels,
            **detection_func_kwargs,
        )

    if detection_overlap is None:
        detection_overlap = target_size_pixels

    return _normalize_overlap(detection_overlap, sdims)


def _run_detection_func(
    block,
    detection_func,
    target_size_pixels,
    detection_func_kwargs,
):
    detections = detection_func(
        block,
        target_size_pixels,
        **detection_func_kwargs,
    )
    if detections.shape != block.shape:
        raise ValueError(
            "detection_func must return a boolean array with the same shape "
            "as its input block."
        )

    if cp is not None and isinstance(detections, cp.ndarray):
        return detections.astype(bool, copy=False)

    return np.asarray(detections, dtype=bool)


def _log_detect_required_overlap(target_size, **_kwargs):
    target_size = tuple(target_size)
    ndim = len(target_size)
    target_size = _normalize_pixel_value(target_size, ndim, "target_size")
    sigma_pixels = tuple(
        max(0.5, size / (2.0 * np.sqrt(ndim))) for size in target_size
    )
    min_distance_pixels = tuple(max(1.0, size / 2.0) for size in target_size)

    return tuple(
        max(1, int(np.ceil(4 * sigma + distance)))
        for sigma, distance in zip(sigma_pixels, min_distance_pixels)
    )


def log_detect(
    image,
    target_size,
    threshold_rel=0.2,
    threshold_abs=None,
    cupy=False,
):
    """
    Detect bright beads in an in-memory array using Laplacian-of-Gaussian.

    Parameters
    ----------
    image : numpy.ndarray or cupy.ndarray
        Input image. Only spatial dimensions should be present.
    target_size : float or sequence[float]
        Expected bead diameter in pixels. A scalar is applied to every axis.
    threshold_rel : float, optional
        Relative threshold applied to the maximum LoG response when
        ``threshold_abs`` is not provided. Defaults to ``0.2``.
    threshold_abs : float or None, optional
        Absolute LoG response threshold. If provided, ``threshold_rel`` is
        ignored.
    cupy : bool, optional
        If ``True`` and CuPy is installed, run the filters on the GPU.

    Returns
    -------
    numpy.ndarray or cupy.ndarray
        Boolean array with the same shape as ``image``. ``True`` pixels are
        local maxima of the LoG response.
    """

    if cupy and cp is None:
        raise ImportError(
            "cupy=True was requested, but CuPy is not installed."
        )

    input_is_cupy = cp is not None and isinstance(image, cp.ndarray)
    if cupy:
        image = cp.asarray(image)

    if cp is not None and isinstance(image, cp.ndarray):
        xp = cp
        ndi = cupyx.scipy.ndimage
        float_dtype = cp.float32
    else:
        xp = np
        ndi = ndimage
        float_dtype = np.float32

    target_size = _normalize_pixel_value(
        target_size,
        image.ndim,
        "target_size",
    )
    sigma_pixels = tuple(
        max(0.5, size / (2.0 * np.sqrt(image.ndim)))
        for size in target_size
    )
    min_distance_pixels = tuple(
        max(1.0, size / 2.0)
        for size in target_size
    )
    max_filter_size = tuple(
        2 * int(np.ceil(distance)) + 1
        for distance in min_distance_pixels
    )

    response = -ndi.gaussian_laplace(
        image.astype(float_dtype, copy=False),
        sigma=sigma_pixels,
        mode="reflect",
    )
    response *= float(np.mean(sigma_pixels)) ** 2

    max_response = ndi.maximum_filter(
        response,
        size=max_filter_size,
        mode="reflect",
    )
    if threshold_abs is None:
        threshold_abs = xp.nanmax(response) * threshold_rel

    detections = (
        (response == max_response)
        & (response > threshold_abs)
        & (response > 0)
    )

    if xp is cp and not input_is_cupy:
        detections = cp.asnumpy(detections)

    return detections


log_detect.required_overlap = _log_detect_required_overlap


def detect_beads(
    msim,
    target_size_physical,
    detection_func=log_detect,
    detection_func_kwargs=None,
    detection_overlap=None,
    segmentation_res_level=None,
    threshold_rel=0.2,
    threshold_abs=None,
    cupy=False,
):
    """
    Detect bright fiducial beads in a multiscale spatial image.

    This function provides the image plumbing: it selects one multiscale level,
    applies ``detection_func`` to the spatial image data, and converts the
    detected pixels to intrinsic physical coordinates. Dask arrays are
    processed with overlap, so large images can be scanned chunk by chunk
    instead of materialising the full detection mask in memory.

    Parameters
    ----------
    msim : multiscale_spatial_image.MultiscaleSpatialImage
        Input multiscale spatial image. If non-spatial dimensions such as
        ``t`` or ``c`` are present, the first coordinate of each is used.
        Select a channel or time point before calling this function to process
        another field.
    target_size_physical : float or dict[str, float]
        Expected bead diameter in physical units. A scalar is applied to every
        spatial dimension; a dict can provide dimension-specific values.
    detection_func : callable, optional
        Function applied to in-memory image blocks. It must accept
        ``(image, target_size_pixels, **kwargs)`` and return a boolean array
        with the same shape as ``image``. By default, ``log_detect`` is used.
    detection_func_kwargs : dict or None, optional
        Additional keyword arguments passed to ``detection_func``.
    detection_overlap : int, sequence[int], dict[str, int], or None, optional
        Pixel overlap used when mapping ``detection_func`` over dask chunks.
        If ``None`` and the detection function exposes a ``required_overlap``
        attribute, that value is used. Otherwise the bead size in pixels is
        used as a conservative default.
    segmentation_res_level : int or None, optional
        Resolution level used for segmentation, e.g. ``0``.
        When ``None``, the coarsest level whose spacing is no larger than one
        quarter of ``target_size_physical`` is used.
    threshold_rel : float, optional
        Passed to ``detection_func`` if it accepts a ``threshold_rel`` keyword.
    threshold_abs : float or None, optional
        Passed to ``detection_func`` if it accepts a ``threshold_abs`` keyword.
    cupy : bool, optional
        Passed to ``detection_func`` if it accepts a ``cupy`` keyword.

    Returns
    -------
    xarray.DataArray
        Detected positions with dimensions ``("point_id", "dim")``. Columns
        follow the image spatial dimension order, and values are in intrinsic
        physical coordinates: ``origin + pixel_index * spacing``. The returned
        array can be passed directly to ``msi_utils.set_point_set`` or
        ``spatial_image_utils.set_point_set``.
    """

    if cupy and cp is None:
        raise ImportError(
            "cupy=True was requested, but CuPy is not installed."
        )

    if segmentation_res_level is not None:
        scale_key = f"scale{segmentation_res_level}"

        if scale_key not in msi_utils.get_sorted_scale_keys(msim):
            raise ValueError(
                f"Resolution level {segmentation_res_level!r} does not exist "
                f"in the multiscale image."
            )
    else:
        sim0 = msi_utils.get_sim_from_msim(msim, scale="scale0")
        sdims0 = si_utils.get_spatial_dims_from_sim(sim0)
        target_size0 = si_utils.normalize_to_spatial_dict(
            target_size_physical, sdims0, "target_size_physical"
        )
        target_spacing = {dim: target_size0[dim] / 4.0 for dim in sdims0}
        res_level = msi_utils.get_res_level_from_spacing(msim, target_spacing)
        scale_key = f"scale{res_level}"
    sim = msi_utils.get_sim_from_msim(msim, scale=scale_key)
    sim = si_utils.get_sim_field(sim)
    sim = si_utils.ensure_dask_backed_dataarray(sim)

    sdims = si_utils.get_spatial_dims_from_sim(sim)
    spacing = si_utils.get_spacing_from_sim(sim)
    origin = si_utils.get_origin_from_sim(sim)
    target_size = si_utils.normalize_to_spatial_dict(
        target_size_physical, sdims, "target_size_physical"
    )
    target_size_pixels = tuple(
        target_size[dim] / spacing[dim] for dim in sdims
    )
    detection_func_kwargs = _prepare_detection_func_kwargs(
        detection_func,
        detection_func_kwargs,
        threshold_rel,
        threshold_abs,
        cupy,
    )
    detection_overlap = _get_detection_overlap(
        detection_func,
        target_size_pixels,
        detection_func_kwargs,
        detection_overlap,
        sdims,
    )

    if isinstance(sim.data, da.Array):
        mask = da.map_overlap(
            _run_detection_func,
            sim.data,
            depth=detection_overlap,
            boundary="reflect",
            trim=True,
            dtype=bool,
            detection_func=detection_func,
            target_size_pixels=target_size_pixels,
            detection_func_kwargs=detection_func_kwargs,
        )
    else:
        mask = _run_detection_func(
            sim.data,
            detection_func=detection_func,
            target_size_pixels=target_size_pixels,
            detection_func_kwargs=detection_func_kwargs,
        )

    point_indices = _compute_point_indices(mask)

    positions = np.empty((len(point_indices), len(sdims)), dtype=float)
    for idim, dim in enumerate(sdims):
        positions[:, idim] = (
            origin[dim] + point_indices[:, idim] * spacing[dim]
        )

    if cupy and cp is not None:
        clear_cupy_memory()

    return xr.DataArray(
        positions,
        dims=["point_id", "dim"],
        coords={"dim": sdims},
        name="position",
        attrs={
            "segmentation_scale": scale_key,
            "target_size_physical": target_size,
            "detection_func": getattr(
                detection_func,
                "__name__",
                str(detection_func),
            ),
        },
    )
