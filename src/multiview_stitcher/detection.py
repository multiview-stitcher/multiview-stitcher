import dask
import dask.array as da
import numpy as np
import xarray as xr
from scipy import ndimage

from multiview_stitcher import msi_utils, fusion, param_utils
from multiview_stitcher import spatial_image_utils as si_utils


def _extract_core_label_centroids(labels, chunk_start, chunk_shape, depth):
    chunk_start = np.asarray(chunk_start, dtype=float)
    chunk_shape = np.asarray(chunk_shape, dtype=float)
    depth = np.asarray(depth, dtype=float)

    label_ids = np.unique(labels)
    label_ids = label_ids[label_ids > 0]
    if len(label_ids) == 0:
        return np.empty((0, labels.ndim), dtype=float)

    centroids = np.asarray(
        ndimage.center_of_mass(labels > 0, labels=labels, index=label_ids),
        dtype=float,
    )
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
        np.concatenate([[0], np.cumsum(np.array(dim_chunks[:-1]) - 2 * depth[idim])])
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


def _log_detect_required_overlap(target_size, **_kwargs):
    ndim = len(target_size)
    sigma_pixels = {
        dim: max(0.5, float(size) / (2.0 * np.sqrt(ndim)))
        for dim, size in target_size.items()
    }
    min_distance_pixels = {
        dim: max(1.0, float(size) / 2.0)
        for dim, size in target_size.items()
    }
    return {
        dim: max(
            1,
            int(np.ceil(4 * sigma_pixels[dim] + min_distance_pixels[dim])),
        )
        for dim in target_size
    }


def log_detect(
    image,
    target_size,
    threshold_rel=0.2,
    threshold_abs=None,
):
    """
    Detect bright beads in an in-memory array using Laplacian-of-Gaussian.

    Parameters
    ----------
    image : numpy.ndarray
        Input image. Only spatial dimensions should be present.
    target_size : float or dict[str, float]
        Expected bead diameter in pixels. A scalar is applied to every axis.
    threshold_rel : float, optional
        Relative threshold applied to the maximum LoG response when
        ``threshold_abs`` is not provided. Defaults to ``0.2``.
    threshold_abs : float or None, optional
        Absolute LoG response threshold. If provided, ``threshold_rel`` is
        ignored.
    Returns
    -------
    numpy.ndarray
        Integer label array with the same shape as ``image``. ``0`` is
        background, and positive labels identify LoG-response local maxima.
    """

    image = np.asarray(image)

    if isinstance(target_size, dict):
        target_size = tuple(float(v) for v in target_size.values())
    elif np.isscalar(target_size):
        target_size = tuple(float(target_size) for _ in range(image.ndim))
    else:
        raise TypeError(
            f"target_size must be a float or dict, got {type(target_size).__name__}."
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

    response = -ndimage.gaussian_laplace(
        image.astype(np.float32, copy=False),
        sigma=sigma_pixels,
        mode="reflect",
    )
    response *= float(np.mean(sigma_pixels)) ** 2

    max_response = ndimage.maximum_filter(
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

    return ndimage.label(detections)[0]


log_detect.required_overlap = _log_detect_required_overlap


def detect_beads(
    msim,
    target_size_physical,
    detection_func=log_detect,
    detection_func_kwargs=None,
    detection_overlap=None,
    segmentation_res_level=None,
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
    target_size_physical : float or dict[str, float]
        Expected bead diameter in physical units. A scalar is applied to every
        spatial dimension; a dict can provide dimension-specific values.
    detection_func : callable, optional
        Function applied to in-memory image blocks. It must accept
        ``image`` and a ``target_size`` keyword argument in pixels, and return
        an integer label array with the same shape as ``image``. ``0`` is
        background and positive values identify objects. By default,
        ``log_detect`` is used.
    detection_func_kwargs : dict or None, optional
        Additional keyword arguments passed to ``detection_func``.
    detection_overlap : int, dict[str, int], or None, optional
        Pixel overlap used when mapping ``detection_func`` over dask chunks.
        If ``None`` and the detection function exposes a ``required_overlap``
        attribute, that value is used. Otherwise the bead size in pixels is
        used as a conservative default.
    segmentation_res_level : int or None, optional
        Resolution level used for segmentation, e.g. ``0``.
        When ``None``, the coarsest level whose spacing is no larger than one
        quarter of ``target_size_physical`` is used.

    Returns
    -------
    xarray.DataArray
        Detected positions with dimensions ``("point_id", "dim")``. Columns
        follow the image spatial dimension order, and values are in intrinsic
        physical coordinates: ``origin + label_centroid * spacing``. The
        returned array can be passed directly to ``msi_utils.set_point_set`` or
        ``spatial_image_utils.set_point_set``.
    """

    # Pick the multiscale resolution level to run detection on.
    # A finer level than necessary wastes memory; a coarser one may miss beads.
    if segmentation_res_level is not None:
        scale_key = f"scale{segmentation_res_level}"

        if scale_key not in msi_utils.get_sorted_scale_keys(msim):
            raise ValueError(
                f"Resolution level {segmentation_res_level!r} does not exist "
                f"in the multiscale image."
            )
    else:
        # Auto-select: find the coarsest level whose voxel spacing is still
        # ≤ target_size / 4, so each bead spans at least ~4 pixels per axis.
        sim0 = msi_utils.get_sim_from_msim(msim, scale="scale0")
        sdims0 = si_utils.get_spatial_dims_from_sim(sim0)
        target_size0 = si_utils.normalize_to_spatial_dict(
            target_size_physical, sdims0, "target_size_physical"
        )
        target_spacing = {dim: target_size0[dim] / 4.0 for dim in sdims0}
        res_level = msi_utils.get_res_level_from_spacing(msim, target_spacing)
        scale_key = f"scale{res_level}"

    # Load the chosen level as a single spatial field (drop t/c if present).
    sim = msi_utils.get_sim_from_msim(msim, scale=scale_key)
    sim = si_utils.get_sim_field(sim)

    sdims = si_utils.get_spatial_dims_from_sim(sim)
    spacing = si_utils.get_spacing_from_sim(sim)
    origin = si_utils.get_origin_from_sim(sim)

    # Convert physical bead size to pixels for detection_func.
    target_size = si_utils.normalize_to_spatial_dict(
        target_size_physical, sdims, "target_size_physical"
    )
    target_size_pixels = {dim: target_size[dim] / spacing[dim] for dim in sdims}

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
    if detection_overlap is None and hasattr(detection_func, "required_overlap"):
        required = detection_func.required_overlap(
            target_size_pixels,
            **detection_func_kwargs,
        )
        detection_overlap = (
            required
            if isinstance(required, dict)
            else dict(zip(sdims, required))
        )
    if detection_overlap is None:
        detection_overlap = target_size_pixels
    detection_overlap = si_utils.normalize_to_spatial_dict(
        detection_overlap, sdims, "detection_overlap"
    )
    detection_overlap = tuple(
        int(np.ceil(detection_overlap[dim])) for dim in sdims
    )


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
        fusion_func=lambda transformed_views, *args, **kwargs: detection_func(
            transformed_views[0], target_size_pixels, **kwargs
        ),
        fusion_func_kwargs=detection_func_kwargs,
        trim_overlap=False,
        overlap_in_pixels={dim: detection_overlap[idim] for idim, dim in enumerate(sdims)},
    ).data.astype(np.int64)

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
            "target_size_physical": target_size,
            "detection_func": getattr(
                detection_func,
                "__name__",
                str(detection_func),
            ),
        },
    )
