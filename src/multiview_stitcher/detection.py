import dask
import numpy as np
import xarray as xr
from scipy import ndimage

from multiview_stitcher import fusion, msi_utils, param_utils
from multiview_stitcher.misc_utils import requires_overlap
from multiview_stitcher import spatial_image_utils as si_utils

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cupyx_ndimage
except ImportError:
    cp = None
    cupyx_ndimage = None


def _is_cupy_array(arr):
    return cp is not None and isinstance(arr, cp.ndarray)


def _as_numpy_array(arr, dtype=None):
    if _is_cupy_array(arr):
        arr = cp.asnumpy(arr)
    elif isinstance(arr, (list, tuple)):
        arr = [_as_numpy_array(item) for item in arr]
    return np.asarray(arr, dtype=dtype)


def _validate_label_array(labels):
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("detection_func must return an integer label array.")


def _extract_core_label_centroids(labels, chunk_start, chunk_shape, depth):
    _validate_label_array(labels)
    is_cupy = _is_cupy_array(labels)
    ndi = cupyx_ndimage if is_cupy else ndimage
    chunk_start = np.asarray(chunk_start, dtype=float)
    chunk_shape = np.asarray(chunk_shape, dtype=float)
    depth = np.asarray(depth, dtype=float)

    label_ids = np.unique(labels)
    label_ids = label_ids[label_ids > 0]
    if len(label_ids) == 0:
        return np.empty((0, labels.ndim), dtype=float)

    # The delayed label blocks may be backed by NumPy or CuPy chunks,
    # depending on the requested fusion backend.
    center_output = ndi.center_of_mass(
        labels, # passing the integer label array prevents cupy from complaining about slow implementation
        labels=labels, index=label_ids
    )
    centroids = _as_numpy_array(center_output, dtype=float)
    if centroids.ndim == 1:
        centroids = centroids[np.newaxis, :]

    core_start = depth
    core_stop = depth + chunk_shape
    keep = np.all((centroids >= core_start) & (centroids < core_stop), axis=1)
    centroids = centroids[keep]
    if len(centroids) == 0:
        return np.empty((0, labels.ndim), dtype=float)

    return chunk_start + centroids - depth


def _compute_point_indices_from_label_blocks(label_blocks, depth):
    delayed_blocks = np.asarray(label_blocks.to_delayed(), dtype=object)
    starts_by_dim = [
        np.concatenate(
            [[0], np.cumsum(np.array(dim_chunks[:-1]) - 2 * depth[idim])]
        )
        for idim, dim_chunks in enumerate(label_blocks.chunks)
    ]

    centroid_tasks = []
    for chunk_index in np.ndindex(label_blocks.numblocks):
        chunk_start = tuple(
            starts_by_dim[idim][ichunk]
            for idim, ichunk in enumerate(chunk_index)
        )
        chunk_shape = tuple(
            label_blocks.chunks[idim][ichunk] - 2 * depth[idim]
            for idim, ichunk in enumerate(chunk_index)
        )
        centroid_tasks.append(
            dask.delayed(_extract_core_label_centroids)(
                delayed_blocks[chunk_index],
                chunk_start,
                chunk_shape,
                depth,
            )
        )

    if len(centroid_tasks) == 0:
        return np.empty((0, label_blocks.ndim), dtype=float)

    point_indices = dask.compute(*centroid_tasks)
    point_indices = [points for points in point_indices if len(points) > 0]
    if len(point_indices) == 0:
        return np.empty((0, label_blocks.ndim), dtype=float)

    return np.concatenate(point_indices, axis=0)


def _normalize_target_size_physical(target_size_physical, ndim):
    if isinstance(target_size_physical, bool):
        raise TypeError(
            "target_size_physical must be a float or dict[str, float]."
        )
    if isinstance(target_size_physical, (int, float, np.integer, np.floating)):
        return tuple(float(target_size_physical) for _ in range(ndim))
    if isinstance(target_size_physical, dict):
        if len(target_size_physical) != ndim or not all(
            isinstance(dim, str) for dim in target_size_physical
        ):
            raise TypeError(
                "target_size_physical must be a float or dict[str, float]."
            )
        return tuple(float(size) for size in target_size_physical.values())
    raise TypeError(
        "target_size_physical must be a float or dict[str, float]."
    )


def _target_size_pixels(target_size_physical, spacing):
    spacing = tuple(float(sp) for sp in spacing)
    target_size_physical = _normalize_target_size_physical(
        target_size_physical, len(spacing)
    )
    return tuple(
        size / sp for size, sp in zip(target_size_physical, spacing)
    )


def _log_detect_required_overlap(kwargs):
    target_size = _target_size_pixels(
        kwargs["target_size_physical"], kwargs["spacing"]
    )
    ndim = len(target_size)
    sigma_pixels = {
        idim: max(0.5, float(size) / (2.0 * np.sqrt(ndim)))
        for idim, size in enumerate(target_size)
    }
    min_distance_pixels = {
        idim: max(1.0, float(size) / 2.0)
        for idim, size in enumerate(target_size)
    }
    return tuple(
        max(
            1,
            int(np.ceil(4 * sigma_pixels[idim] + min_distance_pixels[idim])),
        )
        for idim in range(ndim)
    )


@requires_overlap(_log_detect_required_overlap)
def log_detect(
    image,
    spacing,
    target_size_physical,
    threshold_rel=0.2,
    threshold_abs=None,
    max_neigh_intensity=None,
    max_neigh_sample_size=None,
    max_neigh_sigma=None,
):
    """
    Detect bright beads in an in-memory array using Laplacian-of-Gaussian.

    Parameters
    ----------
    image : numpy.ndarray
        Input image. Only spatial dimensions should be present.
    spacing : tuple[float, ...]
        Pixel spacing for each image axis.
    target_size_physical : float or dict[str, float]
        Expected bead diameter in physical units. A float is applied to every
        axis; a dict can provide axis-specific values.
    threshold_rel : float, optional
        Relative threshold applied to the maximum LoG response when
        ``threshold_abs`` is not provided. Defaults to ``0.2``.
    threshold_abs : float or None, optional
        Absolute LoG response threshold. If provided, ``threshold_rel`` is
        ignored.
    max_neigh_intensity : float or None, optional
        If provided, the minimum pixel intensity in a neighborhood around each
        candidate detection must be less than this value for the candidate to be
        accepted. This can help filter out detections within the interior of bright objects.
    max_neigh_sample_size : float or dict[str, float] or None, optional
        Specifies the size of the neighborhood used to compute the minimum pixel intensity
    max_neigh_sigma : float or dict[str, float] or None, optional
        If provided, intensities are smoothed with a Gaussian filter before computing
        the minimum in the neighborhood. The sigma of the Gaussian filter is specified in physical units.
    Returns
    -------
    numpy.ndarray
        Integer label array with the same shape as ``image``. ``0`` is
        background, and positive labels identify LoG-response local maxima.
    """

    is_cupy = _is_cupy_array(image)
    ndi = cupyx_ndimage if is_cupy else ndimage
    target_size = _target_size_pixels(target_size_physical, spacing)
    if len(target_size) != image.ndim:
        raise ValueError(
            "spacing and target_size_physical must match image.ndim."
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
        image.astype(np.float32, copy=False),
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
        threshold_abs = np.nanmax(response) * threshold_rel

    detections = (
        (response == max_response)
        & (response > threshold_abs)
        & (response > 0)
    )

    if max_neigh_intensity is not None:
        if max_neigh_sigma is not None:

            max_neigh_sigma = _normalize_target_size_physical(
                max_neigh_sigma, image.ndim
            )

            max_neigh_sigma_pixels = tuple(
                si / sp for si, sp in zip(max_neigh_sigma, spacing)
            )

            sample_intensities = ndi.gaussian_filter(
                image.astype(np.float32, copy=False),
                sigma=max_neigh_sigma_pixels,
            )
        else:
            sample_intensities = image

        if max_neigh_sample_size is not None:

            max_neigh_sample_size = _normalize_target_size_physical(
                max_neigh_sample_size, image.ndim
            )
            min_filter_size_physical = max_neigh_sample_size
        else:
            min_filter_size_physical = _normalize_target_size_physical(
                target_size_physical, image.ndim
            )

        min_filter_size= [sip / sp
                for sip, sp in zip(min_filter_size_physical, spacing)]            
            
        min_sample_intensities = ndi.minimum_filter(
            sample_intensities,
            size=min_filter_size,
            mode="reflect",
        )
        detections &= (min_sample_intensities < max_neigh_intensity)

    return ndi.label(detections)[0]


def detect_beads(
    msim,
    detection_func=log_detect,
    detection_func_kwargs=None,
    detection_overlap=None,
    max_detection_spacing=None,
    backend=None,
):
    """
    Detect bright fiducial beads in a multiscale spatial image.

    This function provides the image plumbing: it selects one multiscale level,
    applies ``detection_func`` to the spatial image data, and converts the
    detected label centroids to intrinsic physical coordinates. Dask arrays are
    processed with overlap, so large images can be scanned chunk by chunk
    instead of materialising the full label image in memory.

    Parameters
    ----------
    msim : multiscale_spatial_image.MultiscaleSpatialImage
        Input multiscale spatial image. If non-spatial dimensions such as
        ``t`` or ``c`` are present, the first coordinate of each is used.
        Select a channel or time point before calling this function to process
        another field.
    detection_func : callable, optional
        Function applied to in-memory image blocks. It must accept
        ``image``, ``spacing`` as a tuple of floats, and additional keyword
        arguments, and return an integer label array. ``0`` is background and
        positive values identify objects. By default, ``log_detect`` is used.
    detection_func_kwargs : dict or None, optional
        Additional keyword arguments passed to ``detection_func``.
    detection_overlap : int, dict[str, int], or None, optional
        Pixel overlap used when mapping ``detection_func`` over dask chunks.
        If ``None`` and the detection function exposes a ``required_overlap``
        attribute, that value is used. Otherwise no overlap is used.
    max_detection_spacing : float, dict[str, float], or None, optional
        Maximum spacing used to select the detection resolution level. If
        ``None``, ``scale0`` is used. Otherwise ``get_res_level_from_spacing``
        selects the coarsest resolution level whose spacing does not exceed
        this value.
    backend : {"numpy", "cupy"} or None, optional
        Compute backend to use. If "cupy" and CuPy is available, detection
        is performed on the GPU: the detection function is passed CuPy arrays
        and should return a CuPy array.

    Returns
    -------
    xarray.DataArray
        Detected positions with dimensions ``("point_id", "dim")``. Columns
        follow the image spatial dimension order, and values are in intrinsic
        physical coordinates: ``origin + label_centroid * spacing``. The
        returned array can be passed directly to ``msi_utils.set_point_set`` or
        ``spatial_image_utils.set_point_set``.
    """

    if backend is None:
        backend = "numpy"
    elif backend not in ("numpy", "cupy"):
        raise ValueError(f"Unsupported backend: {backend}")

    if max_detection_spacing is None:
        scale_key = "scale0"
    else:
        sim0 = msi_utils.get_sim_from_msim(msim, scale="scale0")
        sdims0 = si_utils.get_spatial_dims_from_sim(sim0)
        max_detection_spacing = si_utils.normalize_to_spatial_dict(
            max_detection_spacing, sdims0, "max_detection_spacing"
        )
        res_level = msi_utils.get_res_level_from_spacing(
            msim, max_detection_spacing
        )
        scale_key = f"scale{res_level}"

    # Load the chosen level as a single spatial field (drop t/c if present).
    sim = msi_utils.get_sim_from_msim(msim, scale=scale_key)
    sim = si_utils.get_sim_field(sim)

    sdims = si_utils.get_spatial_dims_from_sim(sim)
    spacing = si_utils.get_spacing_from_sim(sim)
    spacing_tuple = tuple(spacing[dim] for dim in sdims)
    origin = si_utils.get_origin_from_sim(sim)

    detection_func_kwargs = (
        dict(detection_func_kwargs)
        if detection_func_kwargs is not None
        else {}
    )

    if detection_overlap is not None and not isinstance(
        detection_overlap, (int, dict)
    ):
        raise TypeError(
            f"detection_overlap must be an int, a dict, or None; "
            f"got {type(detection_overlap).__name__}."
        )

    # Determine how many pixels of overlap are needed between dask chunks so
    # that detection results near chunk boundaries are not clipped.
    if detection_overlap is None and hasattr(
        detection_func, "required_overlap"
    ):
        required = detection_func.required_overlap(
            detection_func_kwargs | {"spacing": spacing_tuple}
        )
        detection_overlap = (
            required
            if isinstance(required, dict)
            else dict(zip(sdims, required))
        )
    if detection_overlap is None:
        detection_overlap = 0
    detection_overlap = si_utils.normalize_to_spatial_dict(
        detection_overlap, sdims, "detection_overlap"
    )
    detection_overlap = tuple(
        int(np.ceil(detection_overlap[dim])) for dim in sdims
    )

    def _detect(transformed_views, spacing, **kwargs):
        labels = detection_func(transformed_views[0], spacing, **kwargs)
        _validate_label_array(labels)
        return labels

    # Set a dummy identity affine on the input SIM so that fusion applies no
    # transformation to the image blocks before passing them to detection_func.
    si_utils.set_sim_affine(
        sim,
        param_utils.identity_transform(len(sim.dims)),
        transform_key="identity",
    )

    # Use fusion.fuse to apply detection_func to each dask block.
    # This allows dealing with both dask-backed and zarr-backed sims,
    # the latter more efficiently.
    label_blocks = fusion.fuse(
        [sim],
        transform_key="identity",
        fusion_func=_detect,
        fusion_func_kwargs=detection_func_kwargs
        | {"spacing": spacing_tuple},
        trim_overlap=False,
        overlap_in_pixels={
            dim: detection_overlap[idim] for idim, dim in enumerate(sdims)
        },
        backend=backend,
        output_on_backend=True,
    ).data.astype(np.int32)

    point_indices = _compute_point_indices_from_label_blocks(
        label_blocks, detection_overlap
    )

    positions = np.empty((len(point_indices), len(sdims)), dtype=float)
    for idim, dim in enumerate(sdims):
        positions[:, idim] = (
            origin[dim] + point_indices[:, idim] * spacing[dim]
        )

    return xr.DataArray(
        positions,
        dims=["point_id", "dim"],
        coords={"dim": sdims},
        name="position",
        attrs={
            "segmentation_scale": scale_key,
            "detection_func": getattr(
                detection_func,
                "__name__",
                str(detection_func),
            ),
        },
    )
