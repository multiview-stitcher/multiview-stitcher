import inspect
import os, shutil
import warnings
from collections.abc import Callable, Sequence
from functools import partial
from itertools import product
from typing import Union
from tqdm import tqdm

import dask
import dask.array as da
import numpy as np
import zarr
import xarray as xr
from dask import delayed
from dask.utils import has_keyword
from dask.array.core import normalize_chunks

from multiview_stitcher import (
    msi_utils,
    mv_graph,
    ngff_utils,
    param_utils,
    misc_utils,
    transformation,
    weights,
)
from multiview_stitcher import spatial_image_utils as si_utils

try:
    import cupy as cp
except ImportError:
    cp = None

import logging

logger = logging.getLogger(__name__)

BoundingBox = dict[str, dict[str, Union[float, int]]]


def max_fusion(
    transformed_views,
):
    """
    Simple pixel-wise maximum fusion.

    Parameters
    ----------
    transformed_views : list of ndarrays
        transformed input views

    Returns
    -------
    ndarray
        Maximum of input views at each pixel
    """
    return np.nanmax(transformed_views, axis=0)


def first_view_fusion(
    transformed_views,
):
    """
    Take each pixel from the first view (in input order) that covers it.

    No blending is performed, so transitions between views are hard seams.

    Parameters
    ----------
    transformed_views : list of ndarrays
        transformed input views, in input order. Pixels not covered by a
        view are NaN.

    Returns
    -------
    ndarray
        Value of the first covering view at each pixel (NaN where no view
        covers the pixel)
    """
    transformed_views = np.asarray(transformed_views)

    # argmax returns the first True along the view axis, i.e. the index of
    # the first view with valid data. Where no view is valid it returns 0,
    # whose value is NaN there anyway.
    first_valid_view = np.argmax(~np.isnan(transformed_views), axis=0)

    # pick, for every pixel, the value of its first valid view
    fused = np.take_along_axis(
        transformed_views, first_valid_view[None], axis=0
    )[0]

    return fused.astype(transformed_views[0].dtype)


def weighted_average_fusion(
    transformed_views,
    blending_weights,
    fusion_weights=None,
):
    """
    Simple weighted average fusion.

    Parameters
    ----------
    transformed_views : list of ndarrays
        transformed input views
    blending_weights : list of ndarrays
        blending weights for each view
    fusion_weights : list of ndarrays, optional
        additional view weights for fusion, e.g. contrast weighted scores.
        By default None.

    Returns
    -------
    ndarray
        Fusion of input views
    """

    if fusion_weights is None:
        # blending_weights are already normalized in fuse_np; skip redundant pass
        additive_weights = blending_weights
    else:
        additive_weights = blending_weights * fusion_weights
        additive_weights = weights.normalize_weights(additive_weights)

    product = transformed_views * additive_weights

    return np.nansum(product, axis=0).astype(transformed_views[0].dtype)


def simple_average_fusion(
    transformed_views,
):
    """
    Simple weighted average fusion.

    Parameters
    ----------
    transformed_views : list of ndarrays
        transformed input views
    blending_weights : list of ndarrays
        blending weights for each view
    fusion_weights : list of ndarrays, optional
        additional view weights for fusion, e.g. contrast weighted scores.
        By default None.

    Returns
    -------
    ndarray
        Fusion of input views
    """

    number_of_valid_views = np.zeros(
        transformed_views[0].shape, dtype=np.float32
    )
    for tv in transformed_views:
        number_of_valid_views = np.nansum(
            [number_of_valid_views, ~np.isnan(tv)], axis=0
        )

    number_of_valid_views[number_of_valid_views == 0] = np.nan

    return (
        np.nansum(transformed_views, axis=0) / number_of_valid_views
    ).astype(transformed_views[0].dtype)


def _fuse_block(
    chunk_params,
    *,
    output_dtype,
    sim_coord_dict,
    sdims,
    fusion_func,
    fusion_func_kwargs,
    weights_func,
    weights_func_kwargs,
    overlap_in_pixels,
    trim_overlap,
    interpolation_order,
    blending_widths,
    shrink_distance,
    backend=None,
    output_on_backend=False,
):
    """
    Read the views relevant for a single output chunk and fuse them.

    Called by ``da.map_blocks`` at dask compute time for zarr-backed input
    (zarr slice reconstruction and fusion happen here, keeping the dask graph
    thin with one task per output block), and for each chunk when fusing
    directly into a zarr array.

    Parameters
    ----------
    chunk_params : dict or np.ndarray
        Parameters for this output chunk as built by ``_get_chunk_params``
        (or a single-element object block containing them), including only
        the relevant input views.
    output_dtype : np.dtype
        Output dtype for zero-filled chunks.
    sim_coord_dict : dict
        Non-spatial coordinate selection for this ns-coord iteration.
    sdims : list of str
        Spatial dimension names.
    fusion_func, fusion_func_kwargs, weights_func, weights_func_kwargs,
    overlap_in_pixels, trim_overlap, interpolation_order, blending_widths,
    shrink_distance, backend, output_on_backend
        Fusion parameters forwarded to ``fuse_np``.

    Returns
    -------
    np.ndarray
        Fused output chunk.
    """
    entry = np.asarray(chunk_params, dtype=object).item()

    output_chunk_bb = entry["output_bb"]
    output_chunk_bb_with_overlap = entry["output_bb_overlap"]
    fuse_planewise = entry["fuse_planewise"]
    output_chunk_bb_result = (
        output_chunk_bb if trim_overlap else output_chunk_bb_with_overlap
    )
    out_shape = tuple(output_chunk_bb_result["shape"][dim] for dim in sdims)

    chunk_views = entry["views"]  # list of (iview, tile_overlap_bb)
    if not chunk_views:
        return np.zeros(out_shape, dtype=output_dtype)

    if "tile_info" in chunk_views[0]:
        # Materialize only the raw zarr region needed for each relevant view.
        sims_slices = [
            si_utils.deserialize_zarr_backed_sim(
                view["tile_info"],
                reconstruct_slice=True,
                overlap_bb=view["tile_overlap_bb"],
                sim_coord_dict=sim_coord_dict,
            )
            for view in chunk_views
        ]
    else:
        # The outer scheduler may be distributed; keep this small inner
        # graph local to the worker processing the chunk.
        sims_slices = dask.compute(
            *[
                _select_view_overlap(
                    view["sim"], sim_coord_dict, view["tile_overlap_bb"], sdims
                )
                for view in chunk_views
            ],
            scheduler="single-threaded",
        )

    return _fuse_block_views(
        sims_slices,
        chunk_views,
        output_chunk_bb_with_overlap=output_chunk_bb_with_overlap,
        fuse_planewise=fuse_planewise,
        fusion_func=fusion_func,
        fusion_func_kwargs=fusion_func_kwargs,
        weights_func=weights_func,
        weights_func_kwargs=weights_func_kwargs,
        overlap_in_pixels=overlap_in_pixels,
        trim_overlap=trim_overlap,
        interpolation_order=interpolation_order,
        blending_widths=blending_widths,
        shrink_distance=shrink_distance,
        backend=backend,
        output_on_backend=output_on_backend,
    )


def _fuse_block_views(
    sims_slices,
    chunk_views,
    *,
    output_chunk_bb_with_overlap,
    fuse_planewise,
    fusion_func,
    fusion_func_kwargs,
    weights_func,
    weights_func_kwargs,
    overlap_in_pixels,
    trim_overlap,
    interpolation_order,
    blending_widths,
    shrink_distance,
    backend=None,
    output_on_backend=False,
):
    """
    Fuse the in-memory view slices relevant for a single output chunk.

    Parameters
    ----------
    sims_slices : list of xr.DataArray
        In-memory slices of the views overlapping the output chunk.
    chunk_views : list of dict
        One entry per slice with keys ``"sparam"`` (the view's affine) and
        ``"view_bb"`` (the stack properties of the full view).
    output_chunk_bb_with_overlap : dict
        Stack properties of the output chunk including the fusion overlap.
    fuse_planewise : bool
        Fuse a single z plane as a 2D problem (needed when a view is itself
        a single plane, whose 3D blending weights are undefined).

    Returns
    -------
    np.ndarray
        Fused output chunk.
    """
    # Fuse a single-plane chunk as a 2D problem (one of its views is a
    # single plane; see _build_spatial_fusion_plan)
    if fuse_planewise:
        sims_slices = [sim.isel(z=0) for sim in sims_slices]
        tmp_params = [
            view["sparam"].sel(x_in=["y", "x", "1"], x_out=["y", "x", "1"])
            for view in chunk_views
        ]
        output_props = mv_graph.project_bb_along_dim(
            output_chunk_bb_with_overlap, dim="z"
        )
        full_view_bbs = [
            mv_graph.project_bb_along_dim(view["view_bb"], dim="z")
            for view in chunk_views
        ]
    else:
        tmp_params = [view["sparam"] for view in chunk_views]
        output_props = output_chunk_bb_with_overlap
        full_view_bbs = [view["view_bb"] for view in chunk_views]

    result = fuse_np(
        sims=sims_slices,
        params=tmp_params,
        output_properties=output_props,
        fusion_func=fusion_func,
        fusion_func_kwargs=fusion_func_kwargs,
        weights_func=weights_func,
        weights_func_kwargs=weights_func_kwargs,
        trim_overlap_in_pixels=overlap_in_pixels if trim_overlap else 0,
        interpolation_order=interpolation_order,
        full_view_bbs=full_view_bbs,
        blending_widths=blending_widths,
        shrink_distance=shrink_distance,
        backend=backend,
        output_on_backend=output_on_backend,
    )

    # Restore the z-axis removed for planewise fusion.
    if fuse_planewise:
        result = result[np.newaxis]

    return result


def _select_view_overlap(sim, sim_coord_dict, tile_overlap_bb, sdims, tol=1e-6):
    """Select the part of a view covered by an overlap bounding box."""
    return sim.sel(
        sim_coord_dict
        | {
            dim: slice(
                tile_overlap_bb["origin"][dim] - tol,
                tile_overlap_bb["origin"][dim]
                + (tile_overlap_bb["shape"][dim] - 1)
                * tile_overlap_bb["spacing"][dim]
                + tol,
            )
            for dim in sdims
        },
        drop=True,
    )


def process_output_chunksize(sims, output_chunksize):

    ndim = si_utils.get_ndim_from_sim(sims[0])
    sdims = si_utils.get_spatial_dims_from_sim(sims[0])

    if output_chunksize is None:
        if si_utils.is_xarray_zarr_backed(sims[0]):
            # For zarr-backed sims, preserve the source chunk grid without
            # converting the full input to dask first.
            preferred_chunks = si_utils._get_preferred_chunks(sims[0])
            output_chunksize = {dim: preferred_chunks[dim] for dim in sdims}
        elif si_utils.is_dask_backed_dataarray(sims[0]):
            # if first tile is a chunked dask array, use its chunksize
            output_chunksize = dict(
                zip(
                    sdims,
                    si_utils._get_backend_data(sims[0]).chunksize[-ndim:],
                )
            )
        else:
            # if first tile is not a chunked dask array, use default chunksize
            # defined in spatial_image_utils.py
            output_chunksize = si_utils.get_default_spatial_chunksizes(ndim)
    elif isinstance(output_chunksize, int):
        output_chunksize = {dim: output_chunksize for dim in sdims}

    return output_chunksize


def _chunks_from_chunk_bbs(chunk_bbs, block_indices, sdims):
    """Return dask chunk sizes matching per-block bounding box shapes."""
    chunks = []

    for idim, dim in enumerate(sdims):
        chunks_for_dim = {}
        for chunk_bb, block_index in zip(chunk_bbs, block_indices):
            chunks_for_dim.setdefault(
                int(block_index[idim]), int(chunk_bb["shape"][dim])
            )
        chunks.append(
            tuple(
                chunks_for_dim[ichunk] for ichunk in range(len(chunks_for_dim))
            )
        )

    return tuple(chunks)


def process_output_stack_properties(
    sims,
    output_spacing=None,
    output_origin=None,
    output_shape=None,
    output_stack_properties=None,
    output_stack_mode="union",
    transform_key=None,
):
    if transform_key is None:
        raise ValueError(
            "transform_key must be provided to determine transformation"
            "parameters for calculating output stack properties."
        )

    params = [
        si_utils.get_affine_from_sim(sim, transform_key=transform_key)
        for sim in sims
    ]

    if output_stack_properties is None:
        if output_spacing is None:
            output_spacing = si_utils.get_spacing_from_sim(sims[0])

        output_stack_properties = calc_fusion_stack_properties(
            sims,
            params=params,
            spacing=output_spacing,
            mode=output_stack_mode,
        )

        if output_origin is not None:
            output_stack_properties["origin"] = output_origin

        if output_shape is not None:
            output_stack_properties["shape"] = output_shape

    return output_stack_properties


def _get_ordered_param_matrices(sparams, sdims):
    """
    Return the affines as one array of shape (n_views, ndim + 1, ndim + 1).

    Rows (``x_in``) and columns (``x_out``) are ordered as ``sdims + ["1"]``,
    so that ``matrices[iview, i, j]`` equals
    ``sparams[iview].sel(x_in=labels[i], x_out=labels[j])``. Reading the
    values once avoids xarray's (comparatively very slow) label-based
    selection for every matrix element.
    """
    labels = list(sdims) + ["1"]
    matrices = np.empty((len(sparams), len(labels), len(labels)), dtype=float)

    for iview, param in enumerate(sparams):
        values = np.asarray(param.data)
        if param.dims.index("x_in") > param.dims.index("x_out"):
            values = values.T
        x_in = param.coords["x_in"].values.tolist()
        x_out = param.coords["x_out"].values.tolist()
        matrices[iview] = values[
            np.ix_(
                [x_in.index(label) for label in labels],
                [x_out.index(label) for label in labels],
            )
        ]

    return matrices


def _prepare_view_geometry(matrices, views_bb, sdims):
    """
    Collect everything the fusion planner needs to know about the views.

    ``matrices`` are the views' affines as returned by
    ``_get_ordered_param_matrices``. Per-view values are stored as arrays so
    that the planner's tests are a few vectorized operations instead of
    Python loops over all views.
    """
    ndim = len(sdims)

    def views_bb_array(key):
        return np.array(
            [[view_bb[key][dim] for dim in sdims] for view_bb in views_bb],
            dtype=float,
        ).reshape((len(views_bb), ndim))

    view_origins = views_bb_array("origin")
    view_spacings = views_bb_array("spacing")

    # Forward-project the view corners (as in
    # mv_graph.get_vertices_from_stack_props) through the affines to get the
    # views' axis-aligned bounding boxes in output (world) space.
    corner_indices = np.array(list(np.ndindex(tuple([2] * ndim))))
    vertices = (
        corner_indices[np.newaxis]
        * (views_bb_array("shape")[:, np.newaxis] - 1)
        * view_spacings[:, np.newaxis]
        + view_origins[:, np.newaxis]
    )
    vertices_output = np.einsum(
        "vij,vkj->vki", matrices[:, :ndim, :ndim], vertices
    ) + matrices[:, np.newaxis, :ndim, ndim]

    return {
        "matrices": matrices,
        "translations": matrices[:, :ndim, ndim],
        "axis_aligned_translation_dims": _get_axis_aligned_translation_dims(
            matrices, sdims
        ),
        "view_origins": view_origins,
        "view_spacings": view_spacings,
        "corners_min": vertices_output.min(axis=1),
        "corners_max": vertices_output.max(axis=1),
    }


def _get_param_from_matrix(matrix, sdims):
    """Return an affine as ``xr.DataArray`` (the inverse of the above)."""
    labels = list(sdims) + ["1"]
    return xr.DataArray(
        matrix,
        dims=["x_in", "x_out"],
        coords={"x_in": labels, "x_out": labels},
    )


def _get_axis_aligned_translation_dims(matrices, sdims, tol=1e-6):
    """
    Return spatial dims that are unaffected except for translation.

    ``matrices`` are the views' affines as returned by
    ``_get_ordered_param_matrices``.
    """
    axis_aligned_dims = []

    for idim, dim in enumerate(sdims):
        other_idims = [iodim for iodim in range(len(sdims)) if iodim != idim]

        # A translation-only dimension keeps its own scale at 1 and has no
        # cross-axis terms in either direction.
        if (
            np.all(np.isclose(matrices[:, idim, idim], 1, atol=tol))
            and np.all(np.isclose(matrices[:, idim, other_idims], 0, atol=tol))
            and np.all(np.isclose(matrices[:, other_idims, idim], 0, atol=tol))
        ):
            axis_aligned_dims.append(dim)

    return axis_aligned_dims


def _get_grid_aligned_translation_dims(
    view_geometry,
    output_stack_properties,
    sdims,
    tol=1e-6,
):
    """Return translation-only dims whose source pixels align to output pixels."""
    axis_aligned_dims = set(view_geometry["axis_aligned_translation_dims"])
    grid_aligned_dims = []

    for idim, dim in enumerate(sdims):
        if dim not in axis_aligned_dims:
            continue

        view_spacings = view_geometry["view_spacings"][:, idim]

        # Grid alignment is only meaningful when source and output pixels have
        # the same spacing along this dimension.
        if not np.all(
            np.isclose(
                output_stack_properties["spacing"][dim],
                view_spacings,
                atol=tol,
            )
        ):
            continue

        # After applying the translation, the output origin must land on a
        # source pixel center. Fractional translations intentionally fail
        # this stricter check but remain axis-aligned translations.
        offsets = (
            output_stack_properties["origin"][dim]
            - view_geometry["translations"][:, idim]
            - view_geometry["view_origins"][:, idim]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            pixel_offsets = offsets / view_spacings

        if np.all(
            (view_spacings != 0)
            & np.isclose(pixel_offsets, np.round(pixel_offsets), atol=tol)
        ):
            grid_aligned_dims.append(dim)

    return grid_aligned_dims


def _get_axis_aligned_translation_overlap(
    target_bb,
    query_bb,
    translation,
    sdims,
    additional_extent_in_pixels=None,
    tol=1e-6,
):
    """
    Return overlap in query coordinates for axis-aligned translations.

    ``translation`` holds the translation components of the view's affine,
    one value per entry of ``sdims``.

    The returned bounding box is an integer source-pixel window covering the
    output chunk back-projected into query coordinates. For fractional
    translations this deliberately over-reads enough source pixels for the
    later interpolation step in ``fuse_np``.
    """
    if additional_extent_in_pixels is None:
        additional_extent_in_pixels = {dim: 0 for dim in sdims}

    overlap_origin = {}
    overlap_shape = {}

    for idim, dim in enumerate(sdims):
        query_spacing = query_bb["spacing"][dim]
        target_spacing = target_bb["spacing"][dim]
        dim_translation = float(translation[idim])

        # Back-project the output pixel-center range into query coordinates.
        # target_bb["shape"] counts centers, so the last sampled center is
        # origin + (shape - 1) * spacing.
        query_min = target_bb["origin"][dim] - dim_translation
        query_max = (
            target_bb["origin"][dim]
            + (int(target_bb["shape"][dim]) - 1) * target_spacing
            - dim_translation
        )
        query_min, query_max = sorted((query_min, query_max))

        additional_extent = (
            additional_extent_in_pixels[dim] * query_spacing
        )
        start_float = (
            query_min
            - additional_extent
            - query_bb["origin"][dim]
        ) / query_spacing
        stop_float = (
            query_max
            + additional_extent
            - query_bb["origin"][dim]
        ) / query_spacing

        # Fractional translations need a covering integer source window; the
        # resampling in fuse_np handles the subpixel offset later.
        start = int(np.floor(start_float + tol))
        stop = int(np.ceil(stop_float - tol)) + 1
        overlap_start = max(start, 0)
        overlap_stop = min(stop, int(query_bb["shape"][dim]))

        if overlap_start >= overlap_stop:
            return None

        overlap_origin[dim] = (
            query_bb["origin"][dim] + overlap_start * query_spacing
        )
        overlap_shape[dim] = overlap_stop - overlap_start

    return {
        "origin": overlap_origin,
        "shape": overlap_shape,
        "spacing": query_bb["spacing"],
    }


def _build_spatial_fusion_plan(
    *,
    view_geometry,
    views_bb,
    output_stack_properties,
    output_chunksize,
    output_chunk_bbs,
    output_chunk_bbs_with_overlap,
    output_chunk_bbs_for_result,
    block_indices,
    overlap_in_pixels,
    trim_overlap,
    interpolation_order,
    sdims,
):
    """
    Determine, for each output chunk, the views overlapping it and the region
    to read from each of them.

    ``view_geometry`` is the output of ``_prepare_view_geometry``.
    """
    axis_aligned_translation_dims = view_geometry[
        "axis_aligned_translation_dims"
    ]
    grid_aligned_translation_dims = _get_grid_aligned_translation_dims(
        view_geometry, output_stack_properties, sdims
    )
    # Axis-aligned translations can use a cheap overlap planner even when they
    # are fractional. Grid-aligned dims are a stricter subset where no
    # interpolation padding is needed.
    use_axis_aligned_translation = (
        set(axis_aligned_translation_dims) == set(sdims)
    )

    # Build a chunk_index -> [tile_indices] mapping by iterating over tiles
    # and using the regular output chunk grid to find overlapping chunks via
    # simple index arithmetic — O(N_tiles * ndim) instead of O(N_tiles * N_chunks).
    #
    # Output chunks form a regular grid: all blocks have the same pixel size
    # per dimension except possibly the last block (which may be smaller).
    _normalized_chunks = normalize_chunks(
        [output_chunksize[dim] for dim in sdims],
        [output_stack_properties["shape"][dim] for dim in sdims],
    )
    n_blocks = np.array([len(c) for c in _normalized_chunks])
    output_origin = np.array(
        [output_stack_properties["origin"][dim] for dim in sdims]
    )
    output_spacing = np.array(
        [output_stack_properties["spacing"][dim] for dim in sdims]
    )

    # Fractional translation dims still use the translation overlap path,
    # but need interpolation support pixels around the source window.
    interpolation_padding_phys = np.where(
        [dim in grid_aligned_translation_dims for dim in sdims],
        0.0,
        float(interpolation_order) * view_geometry["view_spacings"],
    )
    overlap_padding_phys = (
        np.array([overlap_in_pixels[dim] for dim in sdims]) * output_spacing
    )
    padding_phys = interpolation_padding_phys + overlap_padding_phys

    # Map the views' padded bounding boxes to (inclusive) ranges of chunk
    # indices. The first (uniform) block size is sufficient to map any
    # physical coordinate to a chunk index via floor division; clamping to the
    # valid range makes sure the last partial block is included when needed.
    block_shape_phys = (
        np.array([c[0] for c in _normalized_chunks]) * output_spacing
    )
    with np.errstate(invalid="ignore"):
        i_first = np.clip(
            np.floor(
                (view_geometry["corners_min"] - padding_phys - output_origin)
                / block_shape_phys
            ),
            0,
            n_blocks,
        ).astype(np.int64)
        i_last = np.clip(
            np.floor(
                (view_geometry["corners_max"] + padding_phys - output_origin)
                / block_shape_phys
            ),
            -1,
            n_blocks - 1,
        ).astype(np.int64)

    _chunk_to_tiles: dict = {}
    for iview in np.flatnonzero(np.all(i_first <= i_last, axis=1)):
        for chunk_idx in product(
            *(
                range(int(i_first[iview, idim]), int(i_last[iview, idim]) + 1)
                for idim in range(len(sdims))
            )
        ):
            _chunk_to_tiles.setdefault(chunk_idx, []).append(int(iview))

    # Pre-compute per-chunk relevant-view data once and reuse it in both
    # fusion paths. For each output chunk, determine which tiles truly
    # overlap and store their tile-slice bounding boxes.
    # The extent is passed in source pixels; grid-aligned dims can use exact
    # integer slices, while fractional dims need interpolation support.
    additional_extent = {
        dim: (
            0
            if dim in grid_aligned_translation_dims
            else int(interpolation_order)
        )
        for dim in sdims
    }

    per_chunk_entries = []
    for (
        output_chunk_bb,
        output_chunk_bb_with_overlap,
        output_chunk_bb_result,
        block_index,
    ) in zip(
        output_chunk_bbs,
        output_chunk_bbs_with_overlap,
        output_chunk_bbs_for_result,
        block_indices,
    ):
        chunk_views = []
        for iview in _chunk_to_tiles.get(tuple(block_index), []):
            if use_axis_aligned_translation:
                overlap = _get_axis_aligned_translation_overlap(
                    target_bb=output_chunk_bb_with_overlap,
                    query_bb=views_bb[iview],
                    translation=view_geometry["translations"][iview],
                    sdims=sdims,
                    additional_extent_in_pixels=additional_extent,
                )
            else:
                overlap = mv_graph.get_overlap_for_bbs(
                    target_bb=output_chunk_bb_with_overlap,
                    query_bbs=[views_bb[iview]],
                    param=_get_param_from_matrix(
                        np.linalg.inv(view_geometry["matrices"][iview]), sdims
                    ),
                    additional_extent_in_pixels=additional_extent,
                    param_is_inverse=True,
                )[0]
            if overlap is not None:
                chunk_views.append((iview, overlap))
        # A single-plane chunk is fused as a 2D problem only if one of its
        # views is a single plane: the 3D blending weights of such a view are
        # undefined (zero extent in z). Otherwise the 2D weights would ignore
        # the views' z borders and blend the plane differently than a thicker
        # chunk does, so the result would depend on the chunk size.
        fuse_planewise = (
            "z" in grid_aligned_translation_dims
            and output_chunk_bb_with_overlap["shape"].get("z", 2) == 1
            and any(
                views_bb[iview]["shape"]["z"] == 1 for iview, _ in chunk_views
            )
        )
        per_chunk_entries.append(
            {
                "views": chunk_views,
                "output_bb": output_chunk_bb,
                "output_bb_overlap": output_chunk_bb_with_overlap,
                "output_bb_result": output_chunk_bb_result,
                "fuse_planewise": fuse_planewise,
            }
        )

    return {
        "fix_dims": grid_aligned_translation_dims,
        "axis_aligned_translation_dims": axis_aligned_translation_dims,
        "grid_aligned_translation_dims": grid_aligned_translation_dims,
        "per_chunk_entries": per_chunk_entries,
        "uses_axis_aligned_translation": use_axis_aligned_translation,
    }


def _select_params_for_coords(params, params_coord_dict):
    sparams = []
    for param in params:
        indexers = {
            dim: coord
            for dim, coord in params_coord_dict.items()
            if dim in param.dims
        }
        # Label-based selection is slow in xarray; skip it when there is
        # nothing to select.
        sparams.append(param.sel(indexers) if indexers else param)

    return sparams


def _resolve_overlap_and_shrink_distance(
    *,
    overlap_in_pixels,
    output_chunksize,
    fusion_func,
    fusion_func_kwargs,
    weights_func,
    weights_func_kwargs,
    sdims,
):
    """
    Determine the chunk overlap and source shrinkage required for fusion.

    Combines the user-supplied overlap with the requirements declared by the
    weights and fusion functions.

    Returns
    -------
    (overlap_in_pixels, shrink_distance) : (dict[str, int], float or dict)
    """
    overlap_in_pixels = overlap_in_pixels or 0
    # normalize to dict[str, int]
    if not isinstance(overlap_in_pixels, dict):
        overlap_in_pixels = {dim: overlap_in_pixels for dim in sdims}
    shrink_distance = 0
    for func, func_kwargs in [
        (weights_func, weights_func_kwargs),
        (fusion_func, fusion_func_kwargs),
    ]:
        if func is not None and hasattr(func, "required_overlap"):
            # Inject output_chunksize so the overlap can be clamped to it.
            _kwargs_with_chunksize = dict(func_kwargs or {})
            if (
                has_keyword(func, "output_chunksize")
                and output_chunksize is not None
            ):
                _kwargs_with_chunksize.setdefault(
                    "output_chunksize", output_chunksize
                )
            curr_overlap = func.required_overlap(_kwargs_with_chunksize)
            # normalize
            if not isinstance(curr_overlap, dict):
                curr_overlap = {dim: curr_overlap for dim in sdims}
            overlap_in_pixels = {
                dim: max(overlap_in_pixels[dim], curr_overlap[dim])
                for dim in sdims
            }
        if func is not None and hasattr(func, "required_source_shrinkage"):
            shrink_distance = func.required_source_shrinkage(func_kwargs)

    return overlap_in_pixels, shrink_distance


def _add_overlap_to_bb(bb, overlap_in_pixels, sdims):
    """Extend a bounding box by ``overlap_in_pixels`` on each side."""
    return bb | {
        "origin": {
            dim: bb["origin"][dim] - overlap_in_pixels[dim] * bb["spacing"][dim]
            for dim in sdims
        },
        "shape": {
            dim: bb["shape"][dim] + 2 * overlap_in_pixels[dim] for dim in sdims
        },
    }


def _get_chunk_params(entry, *, sparams, views_bb, tile_infos=None, sims=None):
    """
    Assemble what ``_fuse_block`` needs to fuse one output chunk.

    ``entry`` is a per-chunk entry of a spatial fusion plan. Views are
    referenced by their serialized zarr representation (``tile_infos``) if
    available, and by the sims themselves otherwise.
    """
    views = []
    for iview, tile_overlap_bb in entry["views"]:
        view = {
            "tile_overlap_bb": tile_overlap_bb,
            "sparam": sparams[iview],
            "view_bb": views_bb[iview],
        }
        if tile_infos is not None:
            view["tile_info"] = tile_infos[iview]
        else:
            view["sim"] = sims[iview]
        views.append(view)

    return {
        "views": views,
        "output_bb": entry["output_bb"],
        "output_bb_overlap": entry["output_bb_overlap"],
        "fuse_planewise": entry["fuse_planewise"],
    }


def _build_zarr_chunk_params(
    *,
    plan,
    tile_series,
    views_bb,
    block_indices,
    nblocks_per_dim,
):
    chunk_params_np = np.empty(nblocks_per_dim, dtype=object)
    for block_index, entry in zip(block_indices, plan["per_chunk_entries"]):
        chunk_params_np[tuple(block_index)] = _get_chunk_params(
            entry,
            sparams=plan["sparams"],
            views_bb=views_bb,
            tile_infos=tile_series,
        )

    return da.from_array(
        chunk_params_np,
        chunks=(1,) * len(nblocks_per_dim),
        # chunk_params_np is an object array containing nested per-block
        # metadata.  Dask's default deterministic naming tokenizes that object
        # graph, which is expensive and does not buy useful cacheability for
        # this freshly built fusion plan.
        name=False,
    )


def fuse(
    images: list = None,
    transform_key: str = None,
    fusion_func: Callable = weighted_average_fusion,
    fusion_func_kwargs: dict = None,
    weights_func: Callable = None,
    weights_func_kwargs: dict = None,
    output_spacing: dict[str, float] = None,
    output_stack_mode: str = "union",
    output_origin: dict[str, float] = None,
    output_shape: dict[str, int] = None,
    output_stack_properties: BoundingBox = None,
    output_chunksize: Union[int, dict[str, int]] = None,
    overlap_in_pixels: int = None,
    trim_overlap: bool = True,
    interpolation_order: int = 1,
    blending_widths: dict[str, float] = None,
    output_zarr_url: str | None = None,
    zarr_options: dict | None = None,
    batch_options: dict | None = None,
    backend: str | None = None,
    output_on_backend: bool = False,
    sims: list | None = None,
):
    """

    Fuse input images.

    This function fuses all (Z)YX views ("fields") contained in the
    input list of images, which can additionally contain C and T dimensions.
    Inputs may be either all SpatialImages or all MultiscaleSpatialImages.
    When fusing MultiscaleSpatialImages lazily, the returned object is also
    multiscale and each output resolution is fused from a suitable input
    resolution level.

    Parameters
    ----------
    images : list of SpatialImage or MultiscaleSpatialImage
        Input images.
    sims : list of SpatialImage or MultiscaleSpatialImage, optional
        Deprecated alias for ``images``. Kept for backwards compatibility.
    transform_key : str, optional
        Which (extrinsic coordinate system) to use as transformation parameters.
        By default None (intrinsic coordinate system).
    fusion_func : Callable, optional
        Fusion function to be applied. This function receives the following
        inputs (as arrays if applicable): transformed_views, blending_weights, fusion_weights, params.
        By default weighted_average_fusion
    fusion_func_kwargs : dict, optional
    weights_func : Callable, optional
        Function to calculate fusion weights. This function always receives
        transformed_views and may additionally receive params and
        blending_weights when those parameters are declared in its
        signature. It returns (non-normalized) fusion weights for each view.
        By default None.
    weights_func_kwargs : dict, optional
    output_spacing : dict, optional
        Spacing of the fused image for each spatial dimension, by default None
    output_stack_mode : str, optional
        Mode to determine output stack properties. Can be one of
        "union", "intersection", "sample". By default "union"
    output_origin : dict, optional
        Origin of the fused image for each spatial dimension, by default None
    output_shape : dict, optional
        Shape of the fused image for each spatial dimension, by default None
    output_stack_properties : dict, optional
        Dictionary describing the output stack with keys
        'spacing', 'origin', 'shape'. Other output_* are ignored
        if this argument is present.
    output_chunksize : int or dict, optional
        Chunksize of the dask data array of the fused image. If the first tile is a
        chunked dask array or a zarr-backed sim with stored chunk hints, its chunk grid
        is used as the default. Otherwise, the default chunksize defined in
        spatial_image_utils.py is used.
    overlap_in_pixels : int or dict, optional
        Pixel overlap added around each output chunk before fusion, by default
        None.
    trim_overlap : bool, optional
        If True, trim fused chunks back to their requested output chunk size
        after fusion. If False, keep the overlap in each fused chunk returned
        by the lazy fusion graph. By default True.
    output_zarr_url : str or None, optional
        If not None, fuse directly into a Zarr store at this location and do so in batches of chunks,
        with each chunk being processed independently. This allows for efficient memory usage and
        works well for very large datasets (successfully tested ~0.5PB on a macbook).
        When provided, fuse() performs eager fusion and returns a SpatialImage backed by the written store.
        For MultiscaleSpatialImage inputs, the returned object remains multiscale:
        OME-Zarr output returns the written MultiscaleSpatialImage, while plain Zarr output
        returns a single-scale MultiscaleSpatialImage backed by the written store.
    zarr_options: dict, optional
        Additional (dict of) options to pass when creating the Zarr store. Keys:
        - ome_zarr : bool, optional
            If True and output_zarr_url is provided, write a NGFF/OME-Zarr multiscale image under
            "<output_zarr_url>/". Otherwise, the fused array is written directly under output_zarr_url.
        - ngff_version : str, optional
            NGFF version used when ome_zarr=True. Default "0.4".
        - zarr_array_creation_kwargs: dict = None, optional
            Additional keyword arguments to pass when creating the Zarr array.
        - overwrite: bool, by default True
    batch_options : dict, optional
        Options for chunked fusion when output_zarr_url is provided. Keys:
        - batch_func: Callable, optional
            Function to process each batch of fused chunks. Inputs:
            1) a list of block_id(s)
            2) function that performs fusion when passed a given block_id
            By default None, in which case the each block is processed sequentially.
        - n_batch: int
            Number of blocks to process in each batch
            (n_batch>1 only compatible with a provided batch_func). By default 1.
        - batch_func_kwargs: dict, optional
            Additional keyword arguments passed to batch_func.
    backend : str or None, optional
        Compute backend to use for fusion. "numpy" (default) runs on the
        CPU. "cupy" transfers each input chunk to the GPU via
        cupy.asarray, runs resampling, blending-weight computation, and
        the fusion function on the GPU, then returns a NumPy-backed result via
        cupy.asnumpy. Requires CuPy to be installed; raises
        ImportError if CuPy is not available. None is equivalent to
        "numpy".
    output_on_backend : bool, optional
        If True, keep each fused chunk on the selected compute backend for
        lazy, non-Zarr output. By default False, which converts CuPy-backed
        results to NumPy before returning them.
    Returns
    -------
    SpatialImage or MultiscaleSpatialImage
        Fused image.
    """
    if images is None:
        if sims is None:
            raise TypeError(
                "fuse() missing 1 required positional argument: 'images'"
            )

        warnings.warn(
            "The fuse(..., sims=...) parameter is deprecated and will be removed "
            "in a future version. Use images=... instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        images = sims
    elif sims is not None:
        raise TypeError(
            "fuse() got both 'images' and deprecated 'sims'. Use only 'images'."
        )

    if not images:
        raise ValueError("images must contain at least one image.")

    input_is_msim = [msi_utils.is_msim(image) for image in images]
    if any(input_is_msim) and not all(input_is_msim):
        # Mixed inputs would make both scale selection and return type ambiguous.
        raise ValueError(
            "All input images must be of the same kind: either all SpatialImages "
            "or all MultiscaleSpatialImages."
        )

    if all(input_is_msim):
        msims = images

        # Scale 0 defines the finest output geometry; coarser outputs derive from it.
        scale0_sims = [
            msi_utils.get_sim_from_msim(msim, scale="scale0") for msim in msims
        ]

        scale0_output_stack_properties = process_output_stack_properties(
            sims=scale0_sims,
            output_spacing=output_spacing,
            output_origin=output_origin,
            output_shape=output_shape,
            output_stack_properties=output_stack_properties,
            output_stack_mode=output_stack_mode,
            transform_key=transform_key,
        )

        if output_zarr_url is not None:
            # The Zarr path writes one sim, so choose the input level
            # matching that output spacing.
            selected_level_sims = [
                msi_utils.get_sim_from_msim(
                    msim,
                    scale="scale%s"
                    % msi_utils.get_res_level_from_spacing(
                        msim, scale0_output_stack_properties["spacing"]
                    ),
                )
                for msim in msims
            ]
            fused = fuse(
                images=selected_level_sims,
                transform_key=transform_key,
                fusion_func=fusion_func,
                fusion_func_kwargs=fusion_func_kwargs,
                weights_func=weights_func,
                weights_func_kwargs=weights_func_kwargs,
                output_stack_mode=output_stack_mode,
                output_stack_properties=scale0_output_stack_properties,
                output_chunksize=output_chunksize,
                overlap_in_pixels=overlap_in_pixels,
                trim_overlap=trim_overlap,
                interpolation_order=interpolation_order,
                blending_widths=blending_widths,
                output_zarr_url=output_zarr_url,
                zarr_options=zarr_options,
                batch_options=batch_options,
                backend=backend,
            )

            if (zarr_options or {}).get("ome_zarr", False):
                return ngff_utils.read_msim_from_ome_zarr(
                    output_zarr_url,
                    transform_key=(
                        transform_key
                        if transform_key is not None
                        else si_utils.DEFAULT_TRANSFORM_KEY
                    ),
                    # The fusion write path should keep returning a dask-backed
                    # msim even though the general OME-Zarr readers default to
                    # zarr-backed arrays.
                    array_backend="dask",
                )

            return msi_utils.get_msim_from_sim(fused, scale_factors=[])

        res_shapes, _, res_abs_factors = msi_utils.calc_resolution_levels(
            scale0_output_stack_properties["shape"],
        )

        fused_sims = []
        for shape, abs_factors in zip(res_shapes, res_abs_factors):
            # Match the center-of-pixel origin convention used by OME-Zarr
            # output for downsampled levels.
            curr_output_stack_properties = {
                "shape": shape,
                "spacing": {
                    dim: scale0_output_stack_properties["spacing"][dim]
                    * abs_factors[dim]
                    for dim in shape
                },
                "origin": {
                    dim: scale0_output_stack_properties["origin"][dim]
                    + (abs_factors[dim] - 1)
                    * scale0_output_stack_properties["spacing"][dim]
                    / 2
                    for dim in shape
                },
            }
            # Fuse each output level from the coarsest input data that is
            # still fine enough.
            curr_sims = [
                msi_utils.get_sim_from_msim(
                    msim,
                    scale="scale%s"
                    % msi_utils.get_res_level_from_spacing(
                        msim, curr_output_stack_properties["spacing"]
                    ),
                )
                for msim in msims
            ]
            fused_sims.append(
                fuse(
                    images=curr_sims,
                    transform_key=transform_key,
                    fusion_func=fusion_func,
                    fusion_func_kwargs=fusion_func_kwargs,
                    weights_func=weights_func,
                    weights_func_kwargs=weights_func_kwargs,
                    output_stack_mode=output_stack_mode,
                    output_stack_properties=curr_output_stack_properties,
                    output_chunksize=output_chunksize,
                    overlap_in_pixels=overlap_in_pixels,
                    trim_overlap=trim_overlap,
                    interpolation_order=interpolation_order,
                    blending_widths=blending_widths,
                    backend=backend,
                    output_on_backend=output_on_backend,
                )
            )

        # The levels have already been fused; assemble them without further
        # downsampling.
        return msi_utils.get_msim_from_sims(fused_sims)

    sims = images

    # If writing directly to Zarr/OME-Zarr, run chunked fusion path and return eagerly.
    if output_zarr_url is not None:
        # Collect batch options with defaults
        batch_options = batch_options or {}
        batch_func = batch_options.get("batch_func", None)
        n_batch = batch_options.get("n_batch", 1)
        batch_func_kwargs = batch_options.get("batch_func_kwargs", None)

        # Collect zarr options with defaults
        zarr_options = zarr_options or {}
        ome_zarr = zarr_options.get("ome_zarr", False)
        ngff_version = zarr_options.get("ngff_version", "0.4")
        overwrite = zarr_options.get("overwrite", True)
        zarr_array_creation_kwargs = zarr_options.get(
            "zarr_array_creation_kwargs", None
        )

        # Resolve store path for data (OME-Zarr stores scale 0 under "<root>/0")
        store_url = (
            os.path.join(output_zarr_url, "0") if ome_zarr else output_zarr_url
        )

        if overwrite and os.path.exists(output_zarr_url):
            shutil.rmtree(output_zarr_url)
        if ome_zarr:
            # Ensure creation kwargs reflect NGFF version when writing OME-Zarr
            zarr_array_creation_kwargs = (
                ngff_utils.update_zarr_array_creation_kwargs_for_ngff_version(
                    ngff_version, zarr_array_creation_kwargs
                )
            )

        # Build kwargs for per-chunk fuse() calls (exclude zarr-specific args to avoid recursion)
        per_chunk_fuse_kwargs = {
            "images": sims,
            "transform_key": transform_key,
            "fusion_func": fusion_func,
            "fusion_func_kwargs": fusion_func_kwargs,
            "weights_func": weights_func,
            "weights_func_kwargs": weights_func_kwargs,
            "output_spacing": output_spacing,
            "output_stack_mode": output_stack_mode,
            "output_origin": output_origin,
            "output_shape": output_shape,
            "output_stack_properties": output_stack_properties,
            "output_chunksize": output_chunksize,
            "overlap_in_pixels": overlap_in_pixels,
            "trim_overlap": trim_overlap,
            "interpolation_order": interpolation_order,
            "blending_widths": blending_widths,
            "backend": backend,
            # Direct Zarr writes require host arrays, so output_on_backend is
            # intentionally not forwarded into these per-chunk fuse calls.
        }

        # Prepare block fusion and process in batches
        block_fusion_info = prepare_block_fusion(
            store_url,
            fuse_kwargs=per_chunk_fuse_kwargs,
            zarr_array_creation_kwargs=zarr_array_creation_kwargs,
        )
        fuse_chunk = block_fusion_info["func"]
        nblocks = block_fusion_info["nblocks"]
        osp = block_fusion_info["output_stack_properties"]
        osp["shape"] = {dim: int(v) for dim, v in osp["shape"].items()}
        print(f"Fusing {np.prod(nblocks)} blocks in batches of {n_batch}...")
        for batch in tqdm(
            misc_utils.ndindex_batches(nblocks, n_batch),
            total=int(np.ceil(np.prod(nblocks) / n_batch)),
        ):
            if batch_func is None:
                for block_id in batch:
                    fuse_chunk(block_id)
            else:
                batch_func(fuse_chunk, batch, **(batch_func_kwargs or {}))

        # Build SpatialImage from zarr array
        fusion_transform_key = transform_key
        fused = si_utils.get_sim_from_array(
            array=da.from_zarr(store_url),
            dims=list(sims[0].dims),
            transform_key=fusion_transform_key,
            scale=osp["spacing"],
            translation=osp["origin"],
            c_coords=sims[0].coords["c"].values,
            t_coords=sims[0].coords["t"].values,
        )

        # Before the write below, which reads it off the sim.
        ngff_utils.copy_ngff_time_transform(sims[0], fused)

        # If requested, write OME-Zarr metadata
        # and multiscale pyramid
        if ome_zarr:
            ngff_utils.write_sim_to_ome_zarr(
                fused,
                output_zarr_url=output_zarr_url,
                overwrite=False,
                batch_options=batch_options,
                zarr_array_creation_kwargs=zarr_array_creation_kwargs,
                ngff_version=ngff_version,
            )

        return fused

    # Default in-memory fusion path (unchanged)
    output_chunksize = process_output_chunksize(sims, output_chunksize)

    output_stack_properties = process_output_stack_properties(
        sims=sims,
        output_spacing=output_spacing,
        output_origin=output_origin,
        output_shape=output_shape,
        output_stack_properties=output_stack_properties,
        output_stack_mode=output_stack_mode,
        transform_key=transform_key,
    )

    sdims = si_utils.get_spatial_dims_from_sim(sims[0])
    nsdims = si_utils.get_nonspatial_dims_from_sim(sims[0])

    params = [
        si_utils.get_affine_from_sim(sim, transform_key=transform_key)
        for sim in sims
    ]

    # determine overlap from weights/fusion methods and user-supplied value
    overlap_in_pixels, shrink_distance = _resolve_overlap_and_shrink_distance(
        overlap_in_pixels=overlap_in_pixels,
        output_chunksize=output_chunksize,
        fusion_func=fusion_func,
        fusion_func_kwargs=fusion_func_kwargs,
        weights_func=weights_func,
        weights_func_kwargs=weights_func_kwargs,
        sdims=sdims,
    )

    # calculate output chunk bounding boxes
    output_chunk_bbs, block_indices = mv_graph.get_chunk_bbs(
        output_stack_properties, output_chunksize
    )

    # Chunk-grid shape reused to build the zarr-backed per-block param array.
    nblocks_per_dim = tuple(int(x) for x in np.max(block_indices, 0) + 1)

    # add overlap to output chunk bounding boxes
    output_chunk_bbs_with_overlap = [
        _add_overlap_to_bb(output_chunk_bb, overlap_in_pixels, sdims)
        for output_chunk_bb in output_chunk_bbs
    ]

    output_chunk_bbs_for_result = (
        output_chunk_bbs if trim_overlap else output_chunk_bbs_with_overlap
    )

    views_bb = [si_utils.get_stack_properties_from_sim(sim) for sim in sims]

    # All-zarr inputs use map_blocks (thin graph); dask inputs use per-chunk delayed.
    input_is_zarr = all(si_utils.is_xarray_zarr_backed(sim) for sim in sims)

    if input_is_zarr:
        # Precompute lightweight tile representations once (outside the ns_coords loop).
        # Spatial coordinate arrays are excluded — they are rebuilt cheaply from
        # spacing + origin at compute time, avoiding large coord array serialisation.
        # serialize_zarr_backed_sim also captures any dropped-dim selections so
        # that sims pre-filtered with sim_sel_coords are handled correctly.
        tile_series = [si_utils.serialize_zarr_backed_sim(sim) for sim in sims]

    param_dependent_nsdims = [
        dim for dim in nsdims if any(dim in param.dims for param in params)
    ]
    # The spatial plan depends on the non-spatial coordinates only through
    # the affines, so one plan is built per index into the coordinates of the
    # non-spatial dims the affines depend on (e.g. per time point).
    spatial_plans = {}

    merges = []
    for ns_index in np.ndindex(*[sims[0].sizes[dim] for dim in nsdims]):
        ns_coords = tuple(
            sims[0].coords[dim][i] for dim, i in zip(nsdims, ns_index)
        )
        sim_coord_dict = {
            ndsim: ns_coords[i] for i, ndsim in enumerate(nsdims)
        }
        params_coord_dict = {
            dim: sim_coord_dict[dim] for dim in param_dependent_nsdims
        }
        spatial_plan_key = tuple(
            ns_index[nsdims.index(dim)] for dim in param_dependent_nsdims
        )

        if spatial_plan_key not in spatial_plans:
            sparams = _select_params_for_coords(params, params_coord_dict)
            spatial_plans[spatial_plan_key] = _build_spatial_fusion_plan(
                view_geometry=_prepare_view_geometry(
                    _get_ordered_param_matrices(sparams, sdims),
                    views_bb,
                    sdims,
                ),
                views_bb=views_bb,
                output_stack_properties=output_stack_properties,
                output_chunksize=output_chunksize,
                output_chunk_bbs=output_chunk_bbs,
                output_chunk_bbs_with_overlap=output_chunk_bbs_with_overlap,
                output_chunk_bbs_for_result=output_chunk_bbs_for_result,
                block_indices=block_indices,
                overlap_in_pixels=overlap_in_pixels,
                trim_overlap=trim_overlap,
                interpolation_order=interpolation_order,
                sdims=sdims,
            ) | {"sparams": sparams}
        spatial_plan = spatial_plans[spatial_plan_key]
        sparams = spatial_plan["sparams"]
        per_chunk_entries = spatial_plan["per_chunk_entries"]

        if input_is_zarr:
            if "zarr_chunk_params" not in spatial_plan:
                spatial_plan["zarr_chunk_params"] = _build_zarr_chunk_params(
                    plan=spatial_plan,
                    tile_series=tile_series,
                    views_bb=views_bb,
                    block_indices=block_indices,
                    nblocks_per_dim=nblocks_per_dim,
                )
            chunk_params = spatial_plan["zarr_chunk_params"]

            # === zarr path: thin dask graph via map_blocks ===
            fused = da.map_blocks(
                _fuse_block,
                chunk_params,
                chunks=_chunks_from_chunk_bbs(
                    output_chunk_bbs_for_result, block_indices, sdims
                ),
                dtype=sims[0].dtype,
                output_dtype=sims[0].dtype,
                sim_coord_dict=sim_coord_dict,
                sdims=sdims,
                fusion_func=fusion_func,
                fusion_func_kwargs=fusion_func_kwargs,
                weights_func=weights_func,
                weights_func_kwargs=weights_func_kwargs,
                overlap_in_pixels=overlap_in_pixels,
                trim_overlap=trim_overlap,
                interpolation_order=interpolation_order,
                blending_widths=blending_widths,
                shrink_distance=shrink_distance,
                backend=backend,
                output_on_backend=output_on_backend,
            )
        else:
            # === dask path: per-chunk delayed tasks ===
            fused_output_chunks = np.empty(
                np.max(block_indices, 0) + 1, dtype=object
            )

            for block_index, entry in zip(block_indices, per_chunk_entries):
                output_chunk_bb_result = entry["output_bb_result"]
                chunk_views = entry["views"]

                if not chunk_views:
                    fused_output_chunks[tuple(block_index)] = da.zeros(
                        tuple(
                            [
                                output_chunk_bb_result["shape"][dim]
                                for dim in sdims
                            ]
                        ),
                        dtype=sims[0].dtype,
                    )
                    continue

                # The view slices stay lazy: they are computed as part of the
                # dask graph of the fused chunk.
                fused_output_chunk = delayed(_fuse_block_views)(
                    [
                        _select_view_overlap(
                            sims[iview], sim_coord_dict, tile_overlap_bb, sdims
                        )
                        for iview, tile_overlap_bb in chunk_views
                    ],
                    [
                        {"sparam": sparams[iview], "view_bb": views_bb[iview]}
                        for iview, _ in chunk_views
                    ],
                    output_chunk_bb_with_overlap=entry["output_bb_overlap"],
                    fuse_planewise=entry["fuse_planewise"],
                    fusion_func=fusion_func,
                    fusion_func_kwargs=fusion_func_kwargs,
                    weights_func=weights_func,
                    weights_func_kwargs=weights_func_kwargs,
                    overlap_in_pixels=overlap_in_pixels,
                    trim_overlap=trim_overlap,
                    interpolation_order=interpolation_order,
                    blending_widths=blending_widths,
                    shrink_distance=shrink_distance,
                    backend=backend,
                    output_on_backend=output_on_backend,
                )

                fused_output_chunk = da.from_delayed(
                    fused_output_chunk,
                    shape=tuple(
                        [output_chunk_bb_result["shape"][dim] for dim in sdims]
                    ),
                    dtype=sims[0].dtype,
                )

                fused_output_chunks[tuple(block_index)] = fused_output_chunk

            fused = da.block(fused_output_chunks.tolist())

        merge = si_utils.to_spatial_image(
            fused,
            dims=sdims,
            scale=output_stack_properties["spacing"],
            translation=output_stack_properties["origin"],
        )

        merge = merge.expand_dims(nsdims)
        merge = merge.assign_coords(
            {ns_coord.name: [ns_coord.values] for ns_coord in ns_coords}
        )
        merges.append(merge)

    if len(merges) > 1:
        # suppress pandas future warning occuring within xarray.concat
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)

            # if sims are named, combine_by_coord returns a dataset
            res = xr.combine_by_coords([m.rename(None) for m in merges])
    else:
        res = merge

    res = si_utils.get_sim_from_xim(res)
    si_utils.set_sim_affine(
        res,
        param_utils.identity_transform(len(sdims)),
        transform_key,
    )

    # order channels in the same way as first input sim
    # (combine_by_coords may change coordinate order)
    if "c" in res.dims:
        res = res.sel({"c": sims[0].coords["c"].values})

    ngff_utils.copy_ngff_time_transform(sims[0], res)

    return res


def func_ignore_nan_warning(func, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore", message="All-NaN slice encountered"
        )
        warnings.filterwarnings(action="ignore", message="Mean of empty slice")
        return func(*args, **kwargs)


def fuse_np(
    sims: Sequence[Union[xr.DataArray, np.ndarray]],
    params: Sequence[xr.DataArray],
    output_properties: BoundingBox,
    fusion_func: Callable = weighted_average_fusion,
    fusion_func_kwargs: dict = None,
    weights_func: Callable = None,
    weights_func_kwargs: Callable = None,
    trim_overlap_in_pixels: int = 0,
    interpolation_order: int = 1,
    full_view_bbs: Sequence[BoundingBox] = None,
    spacings: Sequence[dict[str, float]] = None,
    origins: Sequence[dict[str, float]] = None,
    blending_widths: dict[float] = None,
    shrink_distance=0,
    backend: str | None = None,
    output_on_backend: bool = False,
):
    """
    Fuse tiles from in-memory slices.

    Parameters
    ----------
    view_data : Sequence[xr.DataArray]
        _description_
    params : Sequence[xr.DataArray]
        _description_
    output_chunk_properties : dict[str, dict[str, Union[int, float]]]
        _description_
    fusion_func : Callable, optional
        _description_, by default weighted_average_fusion
    weights_func : _type_, optional
        _description_, by default None
    weights_func_kwargs : _type_, optional
        _description_, by default None
    trim_overlap_in_pixels : int or dict, optional
        Number of pixels to remove from both sides of each spatial dimension
        after fusion. Pass 0 to keep the full overlapped output chunk. By
        default 0.
    interpolation_order : int, optional
        _description_, by default 1
    full_view_bbs : Sequence[dict[str, dict[str, Union[int, float]]]], optional
        _description_, by default None
    spacings : Sequence[dict[str, float]], optional
        _description_, by default None
    origins : Sequence[dict[str, float]], optional
        _description_, by default None
    output_on_backend : bool, optional
        If True, leave backend-native outputs (for example CuPy arrays) on
        that backend. By default False.

    Returns a delayed object.
    """

    # # convert to xarray.DataArray
    # # it's useful to be able to pass numpy arrays to this function
    # # e.g. being able to apply xr.apply_ufunc
    # for isim in range(len(sims)):
    #     if not isinstance(sims[isim], xr.DataArray):
    #         sims[isim] = si.to_spatial_image(
    #             sims[isim],
    #             dims=si_utils.SPATIAL_DIMS[-sims[isim].ndim :],
    #             scale=spacings[isim],
    #             translation=origins[isim],
    #         )

    if backend == "cupy":
        if cp is None:
            raise ImportError(
                "CuPy is not installed. Install it to use backend='cupy'."
            )
        sims = [
            sim.copy(data=cp.asarray(si_utils._get_backend_data(sim)))
            for sim in sims
        ]

    if has_keyword(fusion_func, "blending_weights") or has_keyword(
        weights_func, "blending_weights"
    ):
        fusion_requires_blending_weights = True
    else:
        fusion_requires_blending_weights = False

    if fusion_func_kwargs is None:
        fusion_func_kwargs = {}
    else:
        # copy to avoid mutating input dict across calls
        fusion_func_kwargs = dict(fusion_func_kwargs)

    if weights_func_kwargs is None:
        weights_func_kwargs = {}
    else:
        # copy to avoid mutating input dict across calls
        weights_func_kwargs = dict(weights_func_kwargs)

    input_dtype = sims[0].dtype

    # A chunk can intersect a view in only one pixel along an axis. Coordinate
    # differences cannot retain the spacing for such singleton slices, so use
    # the known spacing of the full view when it is available.
    if spacings is None:
        spacings = (
            [view_bb["spacing"] for view_bb in full_view_bbs]
            if full_view_bbs is not None
            else [None] * len(sims)
        )

    # Transform input views
    field_ims_t = [
        transformation.transform_sim(
            sim.astype(np.float32),
            np.linalg.inv(param),
            output_stack_properties=output_properties,
            input_spacing=spacing,
            order=interpolation_order,
            cval=np.nan,
        ).data
        for sim, param, spacing in zip(sims, params, spacings)
    ]
    field_ims_t = np.stack(field_ims_t)

    # get blending weights
    if fusion_requires_blending_weights:
        field_ws_t = [
            weights.get_blending_weights(
                target_bb=output_properties,
                source_bb=full_view_bbs[iview],
                affine=params[iview],
                blending_widths=blending_widths,
                shrink_distance=shrink_distance,
                cupy=(backend == "cupy"),
            )
            for iview in range(len(sims))
        ]
        field_ws_t = np.stack(field_ws_t)
        field_ws_t = field_ws_t * ~np.isnan(field_ims_t)
        field_ws_t = weights.normalize_weights(field_ws_t)
    else:
        field_ws_t = None

    fusion_func_kwargs["transformed_views"] = field_ims_t
    if has_keyword(fusion_func, "params"):
        fusion_func_kwargs["params"] = params
    if fusion_requires_blending_weights:
        fusion_func_kwargs["blending_weights"] = field_ws_t
    if (
        has_keyword(fusion_func, "output_spacing")
        and "output_spacing" not in fusion_func_kwargs
    ):
        fusion_func_kwargs["output_spacing"] = output_properties["spacing"]

    # calculate fusion weights if required
    if weights_func is not None and has_keyword(fusion_func, "fusion_weights"):
        weights_func_kwargs["transformed_views"] = field_ims_t
        if has_keyword(weights_func, "params"):
            weights_func_kwargs["params"] = params
        if has_keyword(weights_func, "blending_weights"):
            weights_func_kwargs["blending_weights"] = field_ws_t
        if (
            has_keyword(weights_func, "output_chunksize")
            and "output_chunksize" not in weights_func_kwargs
        ):
            weights_func_kwargs["output_chunksize"] = output_properties[
                "shape"
            ]

        fusion_weights = weights_func(**weights_func_kwargs)
        fusion_func_kwargs["fusion_weights"] = fusion_weights

    fused = func_ignore_nan_warning(
        fusion_func,
        **fusion_func_kwargs,
    )

    # trim overlap
    # normalize to dict[str, int]
    if not isinstance(trim_overlap_in_pixels, dict):
        trim_overlap_in_pixels = {
            dim: trim_overlap_in_pixels
            for dim in output_properties["shape"].keys()
        }

    if any(
        trim_overlap_in_pixels[dim] > 0
        for dim in output_properties["shape"].keys()
    ):
        fused = fused[
            tuple(
                (
                    slice(
                        trim_overlap_in_pixels[dim],
                        -trim_overlap_in_pixels[dim],
                    )
                    if trim_overlap_in_pixels[dim] > 0
                    else slice(None)
                )
                for dim in output_properties["shape"].keys()
            )
        ]

    fused = np.nan_to_num(fused).astype(input_dtype)
    # Keep the historical behavior unless requested otherwise: chunks computed
    # on CuPy are copied back to host memory before they leave fuse_np.
    if (
        cp is not None
        and isinstance(fused, cp.ndarray)
        and not output_on_backend
    ):
        fused = cp.asnumpy(fused)

    # delete references to intermediate arrays to free memory
    del field_ims_t
    if fusion_requires_blending_weights:
        del field_ws_t
    if weights_func is not None and has_keyword(fusion_func, "fusion_weights"):
        del fusion_weights

    del fusion_func_kwargs
    del weights_func_kwargs

    return fused


def calc_fusion_stack_properties(
    sims,
    params,
    spacing,
    mode="union",
):
    """
    Calculate fusion stack properties from input views
    and transformation parameters.

    Parameters
    ----------
    sims : Sequence of SpatialImage
        Input views.
    params : Sequence of xarray.DataArray
        Transformation parameters for each view.
    spacing : ndarray
        Spacing of the output stack.
    mode : str, optional
        'union': output stack entirely contains all (transformed) input views
        'intersection': output stack entirely contains the intersection of
            all (transformed) input views.
        'sample': output stack contains the front (z=0) plane of each input view.
        By default "union".

    Returns
    -------
    dict
        Stack properties (shape, spacing, origin).
    """

    sdims = si_utils.get_spatial_dims_from_sim(sims[0])

    views_props = [
        si_utils.get_stack_properties_from_sim(sim, asarray=False)
        for sim in sims
    ]

    params_ds = xr.Dataset(dict(enumerate(params)))
    param_nsdims = param_utils.get_non_spatial_dims_from_params(params_ds)

    # if present, combine stack properties from multiple non-spatial coordinates
    if len(param_nsdims):
        stack_properties = combine_stack_props(
            [
                calc_stack_properties_from_view_properties_and_params(
                    views_props,
                    [
                        params_ds.sel(
                            {
                                ndsim: ns_coords[i]
                                for i, ndsim in enumerate(param_nsdims)
                            }
                        )
                        .data_vars[ip]
                        .data
                        for ip in range(len(params))
                    ],
                    spacing=spacing,
                    mode=mode,
                )
                for ns_coords in product(
                    *tuple([params_ds.coords[nsdim] for nsdim in param_nsdims])
                )
            ]
        )
    else:
        stack_properties = (
            calc_stack_properties_from_view_properties_and_params(
                views_props,
                [p.data for p in params],
                spacing=spacing,
                mode=mode,
            )
        )

    # return properties in dict form
    stack_properties = {
        k: {dim: v[idim] for idim, dim in enumerate(sdims)}
        for k, v in stack_properties.items()
    }

    return stack_properties


def calc_stack_properties_from_view_properties_and_params(
    views_props,
    params,
    spacing,
    mode="union",
):
    """
    Calculate fusion stack properties.

    Parameters
    ----------
    views_props : list of dict
        Stack properties of input views.
    params : list of xarray.DataArray
        Transformation parameters for each view.
    spacing : ndarray
        Spacing of the output stack.
    mode : str, optional
        'union': output stack entirely contains all (transformed) input views
        'intersection': output stack entirely contains the intersection of
            all (transformed) input views.
        'sample': output stack contains the front (z=0) plane of each input view.
        By default "union".

    Returns
    -------
    dict
        Stack properties (shape, spacing, origin).
    """
    spatial_dims = ["z", "y", "x"][-len(spacing) :]

    # transform into array form
    spacing = np.array([spacing[dim] for dim in spatial_dims])
    views_props = [
        {k: np.array([v[dim] for dim in spatial_dims]) for k, v in vp.items()}
        for vp in views_props
    ]

    spacing = np.array(spacing).astype(float)
    ndim = len(spacing)

    stack_vertices = np.array(list(np.ndindex(tuple([2] * ndim)))).astype(
        float
    )

    if mode == "sample":
        zero_z_face_vertices = stack_vertices[
            np.where(stack_vertices[:, 0] == 1)
        ]
        zero_z_face_vertices[:, 2] = np.mean(
            zero_z_face_vertices[:, 2]
        )  # take mean in x
        transformed_vertices = get_transformed_stack_vertices(
            zero_z_face_vertices, views_props, params
        )
        volume = np.min(np.min(transformed_vertices, 1), 0), np.max(
            np.max(transformed_vertices, 1), 0
        )

    elif mode == "union":
        transformed_vertices = get_transformed_stack_vertices(
            stack_vertices, views_props, params
        )
        volume = np.min(np.min(transformed_vertices, 1), 0), np.max(
            np.max(transformed_vertices, 1), 0
        )

    elif mode == "intersection":
        transformed_vertices = get_transformed_stack_vertices(
            stack_vertices, views_props, params
        )
        volume = np.max(np.min(transformed_vertices, 1), 0), np.min(
            np.max(transformed_vertices, 1), 0
        )

    stack_properties = calc_stack_properties_from_volume(volume, spacing)

    return stack_properties


def combine_stack_props(stack_props_list):
    """
    Combine stack properties from multiple timepoints.

    TODO: This should probably be replaced by simply reusing
    calc_stack_properties_from_view_properties_and_params.

    Parameters
    ----------
    stack_props_list : list of dict

    Returns
    -------
    dict
        Combined stack properties
    """
    combined_stack_props = {}
    combined_stack_props["origin"] = np.min(
        [sp["origin"] for sp in stack_props_list], axis=0
    )
    combined_stack_props["spacing"] = np.min(
        [sp["spacing"] for sp in stack_props_list], axis=0
    )
    # Stack bounds are expressed in pixel-center coordinates. The last valid
    # center is origin + (shape - 1) * spacing; using shape * spacing would add
    # an empty trailing output row/column for fractional translations.
    combined_stack_props["shape"] = (
        np.max(
            [
                np.floor(
                    (
                        sp["origin"]
                        + (sp["shape"] - 1) * sp["spacing"]
                        - combined_stack_props["origin"]
                    )
                    / combined_stack_props["spacing"]
                    + 1e-9
                )
                for sp in stack_props_list
            ],
            axis=0,
        ).astype(np.uint64)
        + 1
    )

    return combined_stack_props


def get_transformed_stack_vertices(
    stack_keypoints, stack_properties_list, params
):
    ndim = len(stack_properties_list[0]["spacing"])
    vertices = np.zeros(
        (len(stack_properties_list), len(stack_keypoints), ndim)
    )
    for iim, sp in enumerate(stack_properties_list):
        # stack_keypoints are 0/1 corner selectors over pixel centers, not over
        # outer pixel edges. This keeps output stack sizing consistent with the
        # coordinates sampled by transform_sim.
        tmp_vertices = stack_keypoints * (
            np.array(sp["shape"]) - 1
        ) * np.array(sp["spacing"]) + np.array(sp["origin"])
        tmp_vertices_transformed = (
            np.dot(params[iim][:ndim, :ndim], tmp_vertices.T).T
            + params[iim][:ndim, ndim]
        )
        vertices[iim] = tmp_vertices_transformed

    return vertices


def calc_stack_properties_from_volume(volume, spacing):
    """
    :param volume: lower and upper edge of final volume (e.g. [edgeLow,edgeHigh] as calculated by calc_final_stack_cube)
    :param spacing: final spacing
    :return: dictionary containing size, origin and spacing of final stack
    """

    origin = volume[0]
    # Convert a pixel-center extent back to a count of sampled centers. The
    # small epsilon keeps values such as 17.999999999 from losing one pixel.
    shape = (
        np.floor((volume[1] - volume[0]) / spacing + 1e-9).astype(np.uint64)
        + 1
    )

    properties_dict = {}
    properties_dict["shape"] = shape
    properties_dict["spacing"] = spacing
    properties_dict["origin"] = origin

    return properties_dict


def get_interpolated_image(
    image: np.ndarray,
    mask: np.ndarray,
    method: str = "nearest",
    fill_value: int = 0,
):
    """

    Currently only 2d!

    :param image: a 2D image
    :param mask: a 2D boolean image, True indicates missing values
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """

    # in case of no known pixels
    if mask.min():
        return image

    from scipy import interpolate

    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = interpolate.griddata(
        (known_x, known_y),
        known_v,
        (missing_x, missing_y),
        method=method,
        fill_value=fill_value,
    )

    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values

    return interp_image


# Keyword arguments of fuse() which control writing to Zarr, and therefore
# are not accepted by prepare_block_fusion.
_FUSE_KWARGS_FOR_ZARR_OUTPUT = frozenset(
    {"output_zarr_url", "zarr_options", "batch_options"}
)

def _fuse_chunk_to_zarr(
    block_id,
    *,
    plan,
    output_zarr_array,
):
    """
    Fuse a single output chunk and write it into an existing Zarr array.

    This is intentionally module-level so distributed schedulers can import and
    pickle it.

    Everything that is the same for all chunks has been computed by
    ``prepare_block_fusion`` and is passed in as ``plan``. Per chunk, the views
    overlapping the chunk are determined with a bounding box test, and only
    those are read, fused and written directly into ``output_zarr_array``.

    Parameters
    ----------
    block_id : sequence of int
        Index of the output chunk, non-spatial dimensions first.
    plan : dict
        Block fusion plan as created by ``prepare_block_fusion``.
    output_zarr_array : zarr.Array
        Array to write the fused chunk into.
    """
    sdims = plan["sdims"]
    nsdims = plan["nsdims"]
    osp = plan["output_stack_properties"]
    fuse_block_kwargs = plan["fuse_block_kwargs"]
    overlap_in_pixels = fuse_block_kwargs["overlap_in_pixels"]

    ns_index = {dim: int(block_id[idim]) for idim, dim in enumerate(nsdims)}
    spatial_chunk_ind = block_id[len(nsdims) :]

    logger.debug(
        "Fusing chunk with block id %s, spatial chunk index %s",
        block_id,
        spatial_chunk_ind,
    )

    # Pixel region of the chunk in the output array. Chunks form a regular
    # grid, of which only the last chunk along each dimension may be smaller.
    chunk_offset = {
        dim: int(b) * plan["output_chunksize"][dim]
        for dim, b in zip(sdims, spatial_chunk_ind)
    }
    chunk_shape = {
        dim: min(
            plan["output_chunksize"][dim],
            int(osp["shape"][dim]) - chunk_offset[dim],
        )
        for dim in sdims
    }

    # The chunk is fused as a small output stack of its own.
    output_chunk_bb = {
        "origin": {
            dim: chunk_offset[dim] * osp["spacing"][dim] + osp["origin"][dim]
            for dim in sdims
        },
        "shape": chunk_shape,
        "spacing": osp["spacing"],
    }
    output_chunk_bb_with_overlap = _add_overlap_to_bb(
        output_chunk_bb, overlap_in_pixels, sdims
    )

    view_geometry = plan["view_geometries"][
        tuple(ns_index[dim] for dim in plan["param_dependent_nsdims"])
    ]

    # Plan the chunk like a (single chunk) output stack: this finds the
    # overlapping views with a cheap bounding box test, and the region to read
    # from each of them.
    spatial_plan = _build_spatial_fusion_plan(
        view_geometry=view_geometry,
        views_bb=plan["views_bb"],
        output_stack_properties=output_chunk_bb,
        output_chunksize=chunk_shape,
        output_chunk_bbs=[output_chunk_bb],
        output_chunk_bbs_with_overlap=[output_chunk_bb_with_overlap],
        output_chunk_bbs_for_result=[
            (
                output_chunk_bb
                if fuse_block_kwargs["trim_overlap"]
                else output_chunk_bb_with_overlap
            )
        ],
        block_indices=[(0,) * len(sdims)],
        overlap_in_pixels=overlap_in_pixels,
        trim_overlap=fuse_block_kwargs["trim_overlap"],
        interpolation_order=fuse_block_kwargs["interpolation_order"],
        sdims=sdims,
    )
    chunk_entry = spatial_plan["per_chunk_entries"][0]

    # Read only the required region of only the overlapping views.
    fused = _fuse_block(
        _get_chunk_params(
            chunk_entry,
            sparams={
                iview: _get_param_from_matrix(
                    view_geometry["matrices"][iview], sdims
                )
                for iview, _ in chunk_entry["views"]
            },
            views_bb=plan["views_bb"],
            tile_infos=plan["tile_infos"],
            sims=plan["sims"],
        ),
        output_dtype=plan["dtype"],
        sim_coord_dict={
            dim: plan["ns_coords"][dim][ns_index[dim]] for dim in nsdims
        },
        sdims=sdims,
        **fuse_block_kwargs,
    )

    # Each fused chunk coincides with one chunk of the output array, so it can
    # be written directly.
    output_zarr_array[
        tuple(
            [slice(ns_index[dim], ns_index[dim] + 1) for dim in nsdims]
            + [
                slice(chunk_offset[dim], chunk_offset[dim] + chunk_shape[dim])
                for dim in sdims
            ]
        )
    ] = fused[(np.newaxis,) * len(nsdims)]

    if cp is not None:
        misc_utils.clear_cupy_memory()

    return


def prepare_block_fusion(
    output_zarr_url: str,
    fuse_kwargs: dict,
    zarr_array_creation_kwargs: dict = None,
    create_output: bool = True,
    overwrite: bool = True,
    verbose: bool = True,
):
    """
    Prepare chunkwise fusion function and number of blocks
    for embarrassingly parallel fusion.

    All work that is the same for every output chunk (output stack
    properties, view bounding boxes, required overlap, ...) is done here, once.
    The returned function then only needs to find and read the views
    overlapping a given chunk, fuse them and write the result.

    Parameters
    ----------
    output_zarr_url : str
        Location of the output Zarr array.
    fuse_kwargs : dict
        Keyword arguments as accepted by ``fuse`` (without those related to
        Zarr output and batching). Not modified.
    zarr_array_creation_kwargs : dict, optional
        Additional keyword arguments to pass when creating the Zarr array.
    create_output : bool, optional
        If True (default), create (and overwrite) the output Zarr array.
        Set to False to attach to an array that already exists, which is what
        additional workers do when the blocks of a single fusion are spread
        over several processes / web workers: one process creates the store,
        the others open it for writing and fuse a disjoint subset of blocks.
    overwrite : bool, optional
        If True (default), replace an existing output array. Set to False for
        stores that cannot enumerate their contents - an HTTP-backed store,
        for instance - where the caller clears the destination itself.
    verbose : bool, optional
        If True (default), print a summary of the output stack.

    Returns
    -------
    dict
        - "func": function fusing and writing the chunk with a given block id
          (index into the chunk grid, non-spatial dimensions first)
        - "nblocks": number of chunks per dimension
        - "output_stack_properties": properties of the fused stack
    """

    fuse_parameters = inspect.signature(fuse).parameters
    unexpected_kwargs = set(fuse_kwargs) - (
        set(fuse_parameters) - _FUSE_KWARGS_FOR_ZARR_OUTPUT
    )
    if unexpected_kwargs:
        raise TypeError(
            "prepare_block_fusion() got unexpected fuse_kwargs: "
            f"{sorted(unexpected_kwargs)}"
        )

    # Fill in the defaults of fuse() (without modifying the passed dict).
    fuse_kwargs = {
        name: parameter.default for name, parameter in fuse_parameters.items()
    } | fuse_kwargs

    sims = fuse_kwargs["images"]
    if sims is None:
        sims = fuse_kwargs["sims"]

    transform_key = fuse_kwargs["transform_key"]

    output_stack_properties = process_output_stack_properties(
        sims=sims,
        output_stack_properties=fuse_kwargs["output_stack_properties"],
        output_spacing=fuse_kwargs["output_spacing"],
        output_origin=fuse_kwargs["output_origin"],
        output_shape=fuse_kwargs["output_shape"],
        output_stack_mode=fuse_kwargs["output_stack_mode"],
        transform_key=transform_key,
    )

    output_chunksize = process_output_chunksize(
        sims, fuse_kwargs["output_chunksize"]
    )

    dims = sims[0].dims
    nsdims = si_utils.get_nonspatial_dims_from_sim(sims[0])
    sdims = si_utils.get_spatial_dims_from_sim(sims[0])
    ns_shape = {dim: len(sims[0].coords[dim]) for dim in nsdims}

    full_output_shape = [ns_shape[dim] for dim in nsdims] + [
        output_stack_properties["shape"][dim] for dim in sdims
    ]
    full_output_chunksize = [
        1,
    ] * len(
        nsdims
    ) + [int(output_chunksize[dim]) for dim in sdims]

    normalized_chunks = normalize_chunks(
        shape=full_output_shape, chunks=full_output_chunksize
    )

    if verbose:
        print("Fusing into an output stack:")
        print(
            "- shape: ",
            {
                dim: (
                    int(output_stack_properties["shape"][dim])
                    if dim in sdims
                    else ns_shape[dim]
                )
                for dim in dims
            },
        )
        print(
            "- spacing: ",
            {
                k: float(v)
                for k, v in output_stack_properties["spacing"].items()
            },
        )
        print(
            "- origin: ",
            {
                k: float(v)
                for k, v in output_stack_properties["origin"].items()
            },
        )
        # print(f"- chunksize: {fuse_kwargs.get('output_chunksize', None)}")

    if create_output:
        # Create the Zarr array store on disk
        output_zarr_array = zarr.create(
            shape=[int(i) for i in full_output_shape],
            chunks=[int(i) for i in full_output_chunksize],
            # Use xarray's metadata-level dtype. Accessing ``.data`` can
            # materialize zarr-backed xarray arrays.
            dtype=sims[0].dtype,
            store=output_zarr_url,  # The path to the directory where the store will be created
            overwrite=overwrite,
            **(
                zarr_array_creation_kwargs
                if zarr_array_creation_kwargs is not None
                else {}
            ),
        )
    else:
        # Attach to an array created by another process / worker.
        output_zarr_array = zarr.open_array(output_zarr_url, mode="r+")

    # --- Everything below is shared by all chunks and computed only once ---

    func_kwargs = {
        key: fuse_kwargs[key]
        for key in [
            "fusion_func",
            "fusion_func_kwargs",
            "weights_func",
            "weights_func_kwargs",
        ]
    }

    overlap_in_pixels, shrink_distance = _resolve_overlap_and_shrink_distance(
        overlap_in_pixels=fuse_kwargs["overlap_in_pixels"],
        output_chunksize=output_chunksize,
        sdims=sdims,
        **func_kwargs,
    )

    if all(si_utils.is_xarray_zarr_backed(sim) for sim in sims):
        # Keep task payloads compact while still passing live zarr.Array
        # objects for interoperability: workers rebuild the required region
        # of a tile from this lightweight representation.
        task_sims = None
        tile_infos = []
        views_bb = []
        for sim in sims:
            tile_info = si_utils.serialize_zarr_backed_sim(sim)
            # Tiles are rebuilt from their origin and spacing, and a sim's
            # spacing is read back from its coordinates. Because of floating
            # point rounding the latter can differ from the stored spacing in
            # the last digit, so plan (and read) using the properties of the
            # rebuilt tile.
            view_bb = si_utils.get_stack_properties_from_sim(
                si_utils.deserialize_zarr_backed_sim(tile_info)
            )
            tile_info["spacing"] = view_bb["spacing"]
            # Affines are passed separately (see ``params``).
            tile_info.pop("transforms", None)
            tile_infos.append(tile_info)
            views_bb.append(view_bb)
    else:
        task_sims = sims
        tile_infos = None
        views_bb = [
            si_utils.get_stack_properties_from_sim(sim) for sim in sims
        ]

    # The view geometry depends on the chunk only through the affines, which
    # may depend on non-spatial coordinates (e.g. time). Compute it for each
    # of these coordinates. The plan is sent along with every chunk, so the
    # geometry holds the affines as arrays rather than as (much slower to
    # pickle) xarray objects.
    params = [
        si_utils.get_affine_from_sim(sim, transform_key=transform_key)
        for sim in sims
    ]
    param_dependent_nsdims = [
        dim for dim in nsdims if any(dim in param.dims for param in params)
    ]
    view_geometries = {
        ns_index: _prepare_view_geometry(
            _get_ordered_param_matrices(
                _select_params_for_coords(
                    params,
                    {
                        dim: sims[0].coords[dim].values[i]
                        for dim, i in zip(param_dependent_nsdims, ns_index)
                    },
                ),
                sdims,
            ),
            views_bb,
            sdims,
        )
        for ns_index in np.ndindex(
            *[ns_shape[dim] for dim in param_dependent_nsdims]
        )
    }

    plan = {
        "sdims": sdims,
        "nsdims": nsdims,
        "dtype": sims[0].dtype,
        # coordinate values along the non-spatial dimensions
        "ns_coords": {dim: sims[0].coords[dim].values for dim in nsdims},
        "output_stack_properties": output_stack_properties,
        "output_chunksize": {dim: int(output_chunksize[dim]) for dim in sdims},
        # input views
        "sims": task_sims,
        "tile_infos": tile_infos,
        "views_bb": views_bb,
        # view geometry for each index into the coordinates of the
        # non-spatial dims the affines depend on
        "view_geometries": view_geometries,
        "param_dependent_nsdims": param_dependent_nsdims,
        # keyword arguments of _fuse_block
        "fuse_block_kwargs": func_kwargs
        | {
            "overlap_in_pixels": overlap_in_pixels,
            "shrink_distance": shrink_distance,
            "trim_overlap": fuse_kwargs["trim_overlap"],
            "interpolation_order": fuse_kwargs["interpolation_order"],
            "blending_widths": fuse_kwargs["blending_widths"],
            "backend": fuse_kwargs["backend"],
        },
    }

    fuse_chunk = partial(
        _fuse_chunk_to_zarr,
        plan=plan,
        output_zarr_array=output_zarr_array,
    )

    nblocks = [len(nc) for nc in normalized_chunks]

    return {
        "func": fuse_chunk,
        "nblocks": nblocks,
        "output_stack_properties": output_stack_properties,
    }


def fuse_to_zarr(*args, **kwargs):
    """
    Deprecated: use fuse(..., output_zarr_url=...) instead.
    """
    warnings.warn(
        "fuse_to_zarr() is deprecated. Use fuse(..., output_zarr_url=<path>) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    raise RuntimeError(
        "fuse_to_zarr() is deprecated. Please call fuse(..., output_zarr_url=<path>) instead."
    )


def fuse_to_multiscale_ome_zarr(*args, **kwargs):
    """
    Deprecated: use fuse(..., output_zarr_url=..., zarr_options={'ome_zarr': True}) instead.
    """
    warnings.warn(
        "fuse_to_multiscale_ome_zarr() is deprecated. Use fuse(..., output_zarr_url=<path>, zarr_options={'ome_zarr': True}) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    raise RuntimeError(
        "fuse_to_multiscale_ome_zarr() is deprecated. Please call fuse(..., output_zarr_url=<path>, zarr_options={'ome_zarr': True}) instead."
    )
