import itertools
import os, shutil
import warnings
from collections.abc import Callable, Sequence
from itertools import product
from typing import Union
from tqdm import tqdm

import dask.array as da
import numpy as np
import zarr
import spatial_image as si
import xarray as xr
from dask import delayed
from dask.utils import has_keyword
from dask.array.core import normalize_chunks
from dask import config as dask_config

from multiview_stitcher import (
    mv_graph,
    ngff_utils,
    param_utils,
    misc_utils,
    transformation,
    weights,
)
from multiview_stitcher import spatial_image_utils as si_utils

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


def process_output_chunksize(sims, output_chunksize):

    ndim = si_utils.get_ndim_from_sim(sims[0])
    sdims = si_utils.get_spatial_dims_from_sim(sims[0])

    if output_chunksize is None:
        if isinstance(sims[0].data, da.Array):
            # if first tile is a chunked dask array, use its chunksize
            output_chunksize = dict(zip(sdims, sims[0].data.chunksize[-ndim:]))
        else:
            # if first tile is not a chunked dask array, use default chunksize
            # defined in spatial_image_utils.py
            output_chunksize = si_utils.get_default_spatial_chunksizes(ndim)
    elif isinstance(output_chunksize, int):
        output_chunksize = {dim: output_chunksize for dim in sdims}

    return output_chunksize


def process_output_stack_properties(
    sims,
    output_spacing,
    output_origin,
    output_shape,
    output_stack_properties,
    output_stack_mode,
    transform_key,
):
    
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


def fuse(
    sims: list,
    transform_key: str = None,
    fusion_func: Callable = weighted_average_fusion,
    fusion_method_kwargs: dict = None,
    weights_func: Callable = None,
    weights_func_kwargs: dict = None,
    output_spacing: dict[str, float] = None,
    output_stack_mode: str = "union",
    output_origin: dict[str, float] = None,
    output_shape: dict[str, int] = None,
    output_stack_properties: BoundingBox = None,
    output_chunksize: Union[int, dict[str, int]] = None,
    overlap_in_pixels: int = None,
    interpolation_order: int = 1,
    blending_widths: dict[str, float] = None,
    output_zarr_url: str | None = None,
    zarr_options: dict | None = None,
    batch_options: dict | None = None,
):
    """

    Fuse input views.

    This function fuses all (Z)YX views ("fields") contained in the
    input list of images, which can additionally contain C and T dimensions.

    Parameters
    ----------
    sims : list of SpatialImage
        Input views.
    transform_key : str, optional
        Which (extrinsic coordinate system) to use as transformation parameters.
        By default None (intrinsic coordinate system).
    fusion_func : Callable, optional
        Fusion function to be applied. This function receives the following
        inputs (as arrays if applicable): transformed_views, blending_weights, fusion_weights, params.
        By default weighted_average_fusion
    fusion_method_kwargs : dict, optional
    weights_func : Callable, optional
        Function to calculate fusion weights. This function receives the
        following inputs: transformed_views (as spatial images), params.
        It returns (non-normalized) fusion weights for each view.
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
        Chunksize of the dask data array of the fused image. If the first tile is a chunked dask array,
        its chunksize is used as the default. If the first tile is not a chunked dask array,
        the default chunksize defined in spatial_image_utils.py is used.
    output_zarr_url : str or None, optional
        If not None, fuse directly into a Zarr store at this location and do so in batches of chunks,
        with each chunk being processed independently. This allows for efficient memory usage and
        works well for very large datasets (successfully tested ~0.5PB on a macbook).
        When provided, fuse() performs eager fusion and returns a SpatialImage backed by the written store.
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
            Number of blocks to process in each batch, by default 1
        - batch_func_kwargs: dict, optional
            Additional keyword arguments passed to batch_func.
    Returns
    -------
    SpatialImage
        Fused image.
    """
    # If writing directly to Zarr/OME-Zarr, run chunked fusion path and return eagerly.
    if output_zarr_url is not None:
        # Collect batch options with defaults
        batch_options = batch_options or {}
        batch_func = batch_options.get("batch_func", None)
        n_batch = batch_options.get("n_batch", 1)
        batch_func_kwargs = batch_options.get("batch_func_kwargs", None)
        zarr_array_creation_kwargs = batch_options.get("zarr_array_creation_kwargs", None)

        # Collect zarr options with defaults
        zarr_options = zarr_options or {}
        ome_zarr = zarr_options.get("ome_zarr", False)
        ngff_version = zarr_options.get("ngff_version", "0.4")
        overwrite = zarr_options.get("overwrite", True)

        # Resolve store path for data (OME-Zarr stores scale 0 under "<root>/0")
        store_url = os.path.join(output_zarr_url, "0") if ome_zarr else output_zarr_url

        if overwrite and os.path.exists(store_url):
            shutil.rmtree(store_url)
        if ome_zarr:
            # Ensure creation kwargs reflect NGFF version when writing OME-Zarr
            zarr_array_creation_kwargs = ngff_utils.update_zarr_array_creation_kwargs_for_ngff_version(
                ngff_version, zarr_array_creation_kwargs
            )

        # Build kwargs for per-chunk fuse() calls (exclude zarr-specific args to avoid recursion)
        per_chunk_fuse_kwargs = {
            "sims": sims,
            "transform_key": transform_key,
            "fusion_func": fusion_func,
            "fusion_method_kwargs": fusion_method_kwargs,
            "weights_func": weights_func,
            "weights_func_kwargs": weights_func_kwargs,
            "output_spacing": output_spacing,
            "output_stack_mode": output_stack_mode,
            "output_origin": output_origin,
            "output_shape": output_shape,
            "output_stack_properties": output_stack_properties,
            "output_chunksize": output_chunksize,
            "overlap_in_pixels": overlap_in_pixels,
            "interpolation_order": interpolation_order,
            "blending_widths": blending_widths,
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

        print(f'Fusing {np.prod(nblocks)} blocks in batches of {n_batch}...')
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

        # If requested, write OME-Zarr metadata
        # and multiscale pyramid
        if ome_zarr:
            ngff_utils.write_sim_to_ome_zarr(
                fused,
                output_zarr_url=output_zarr_url,
                overwrite=False,
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

    # determine overlap from weights method
    # (soon: fusion methods will also require overlap)
    overlap_in_pixels = 0
    if weights_func is not None:
        overlap_in_pixels = np.max(
            [
                overlap_in_pixels,
                weights.calculate_required_overlap(
                    weights_func, weights_func_kwargs
                ),
            ]
        )

    # calculate output chunk bounding boxes
    output_chunk_bbs, block_indices = mv_graph.get_chunk_bbs(
        output_stack_properties, output_chunksize
    )

    # add overlap to output chunk bounding boxes
    output_chunk_bbs_with_overlap = [
        output_chunk_bb
        | {
            "origin": {
                dim: output_chunk_bb["origin"][dim]
                - overlap_in_pixels * output_stack_properties["spacing"][dim]
                for dim in sdims
            }
        }
        | {
            "shape": {
                dim: output_chunk_bb["shape"][dim] + 2 * overlap_in_pixels
                for dim in sdims
            }
        }
        for output_chunk_bb in output_chunk_bbs
    ]

    views_bb = [si_utils.get_stack_properties_from_sim(sim) for sim in sims]

    merges = []
    for ns_coords in itertools.product(
        *tuple([sims[0].coords[nsdim] for nsdim in nsdims])
    ):
        sim_coord_dict = {
            ndsim: ns_coords[i] for i, ndsim in enumerate(nsdims)
        }
        params_coord_dict = {
            ndsim: ns_coords[i]
            for i, ndsim in enumerate(nsdims)
            if ndsim in params[0].dims
        }

        # ssims = [sim.sel(sim_coord_dict) for sim in sims]
        sparams = [param.sel(params_coord_dict) for param in params]

        # should this be done within the loop over output chunks?
        fix_dims = []
        for dim in sdims:
            other_dims = [odim for odim in sdims if odim != dim]
            if (
                any((param.sel(x_in=dim, x_out=dim) - 1) for param in sparams)
                or any(
                    any(param.sel(x_in=dim, x_out=other_dims))
                    for param in sparams
                )
                or any(
                    any(param.sel(x_in=other_dims, x_out=dim))
                    for param in sparams
                )
                or any(
                    output_stack_properties["spacing"][dim]
                    - views_bb[iview]["spacing"][dim]
                    for iview in range(len(sims))
                )
                or any(
                    float(
                        output_stack_properties["origin"][dim]
                        - param.sel(x_in=dim, x_out="1")
                    )
                    % output_stack_properties["spacing"][dim]
                    for param in sparams
                )
            ):
                continue
            fix_dims.append(dim)

        fused_output_chunks = np.empty(
            np.max(block_indices, 0) + 1, dtype=object
        )

        for output_chunk_bb, output_chunk_bb_with_overlap, block_index in zip(
            output_chunk_bbs, output_chunk_bbs_with_overlap, block_indices
        ):
            # calculate relevant slices for each output chunk
            # this is specific to each non spatial coordinate
            views_overlap_bb = [
                mv_graph.get_overlap_for_bbs(
                    target_bb=output_chunk_bb_with_overlap,
                    query_bbs=[view_bb],
                    param=sparams[iview],
                    additional_extent_in_pixels={
                        dim: 0 if dim in fix_dims else int(interpolation_order)
                        for dim in sdims
                    },
                )[0]
                for iview, view_bb in enumerate(views_bb)
            ]

            # append to output
            relevant_view_indices = np.where(
                [
                    view_overlap_bb is not None
                    for view_overlap_bb in views_overlap_bb
                ]
            )[0]

            if not len(relevant_view_indices):
                fused_output_chunks[tuple(block_index)] = da.zeros(
                    tuple([output_chunk_bb["shape"][dim] for dim in sdims]),
                    dtype=sims[0].dtype,
                )
                continue

            tol = 1e-6
            sims_slices = [
                sims[iview].sel(
                    sim_coord_dict
                    | {
                        dim: slice(
                            views_overlap_bb[iview]["origin"][dim] - tol,
                            views_overlap_bb[iview]["origin"][dim]
                            + (views_overlap_bb[iview]["shape"][dim] - 1)
                            * views_overlap_bb[iview]["spacing"][dim]
                            + tol,
                        )
                        for dim in sdims
                    },
                    drop=True,
                )
                for iview in relevant_view_indices
            ]

            # determine whether to fuse plany by plane
            #  to avoid weighting edge artifacts
            # fuse planewise if:
            # - z dimension is present
            # - params don't affect z dimension
            # - shape in z dimension is 1 (i.e. only one plane)
            # (the last criterium above could be dropped if we find a way
            # (to propagate metadata through xr.apply_ufunc)

            if (
                "z" in fix_dims
                and output_chunk_bb_with_overlap["shape"]["z"] == 1
            ):
                fuse_planewise = True

                sims_slices = [sim.isel(z=0) for sim in sims_slices]
                tmp_params = [
                    sparams[iview].sel(
                        x_in=["y", "x", "1"],
                        x_out=["y", "x", "1"],
                    )
                    for iview in relevant_view_indices
                ]

                output_chunk_bb_with_overlap = mv_graph.project_bb_along_dim(
                    output_chunk_bb_with_overlap, dim="z"
                )

                full_view_bbs = [
                    mv_graph.project_bb_along_dim(views_bb[iview], dim="z")
                    for iview in relevant_view_indices
                ]

            else:
                fuse_planewise = False
                tmp_params = [
                    sparams[iview] for iview in relevant_view_indices
                ]
                full_view_bbs = [
                    views_bb[iview] for iview in relevant_view_indices
                ]

            fused_output_chunk = delayed(
                lambda append_leading_axis, **kwargs: fuse_np(**kwargs)[
                    np.newaxis
                ]
                if append_leading_axis
                else fuse_np(**kwargs),
            )(
                append_leading_axis=fuse_planewise,
                sims=sims_slices,
                params=tmp_params,
                output_properties=output_chunk_bb_with_overlap,
                fusion_func=fusion_func,
                fusion_method_kwargs=fusion_method_kwargs,
                weights_func=weights_func,
                weights_func_kwargs=weights_func_kwargs,
                trim_overlap_in_pixels=overlap_in_pixels,
                interpolation_order=1,
                full_view_bbs=full_view_bbs,
                blending_widths=blending_widths,
            )

            fused_output_chunk = da.from_delayed(
                fused_output_chunk,
                shape=tuple([output_chunk_bb["shape"][dim] for dim in sdims]),
                dtype=sims[0].dtype,
            )

            fused_output_chunks[tuple(block_index)] = fused_output_chunk

        fused = da.block(fused_output_chunks.tolist())

        merge = si.to_spatial_image(
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
    fusion_method_kwargs: dict = None,
    weights_func: Callable = None,
    weights_func_kwargs: Callable = None,
    trim_overlap_in_pixels: int = 0,
    interpolation_order: int = 1,
    full_view_bbs: Sequence[BoundingBox] = None,
    spacings: Sequence[dict[str, float]] = None,
    origins: Sequence[dict[str, float]] = None,
    blending_widths: dict[float] = None,
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
    overlap_in_pixels : int, optional
        _description_, by default None
    interpolation_order : int, optional
        _description_, by default 1
    full_view_bbs : Sequence[dict[str, dict[str, Union[int, float]]]], optional
        _description_, by default None
    spacings : Sequence[dict[str, float]], optional
        _description_, by default None
    origins : Sequence[dict[str, float]], optional
        _description_, by default None

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

    if has_keyword(fusion_func, "blending_weights") or has_keyword(
        weights_func, "blending_weights"
    ):
        fusion_requires_blending_weights = True
    else:
        fusion_requires_blending_weights = False

    if fusion_method_kwargs is None:
        fusion_method_kwargs = {}

    if weights_func_kwargs is None:
        weights_func_kwargs = {}

    input_dtype = sims[0].dtype
    ndim = si_utils.get_ndim_from_sim(sims[0])
    si_utils.get_spatial_dims_from_sim(sims[0])

    # Transform input views
    field_ims_t = [
        transformation.transform_sim(
            sim.astype(float),
            np.linalg.inv(param),
            output_stack_properties=output_properties,
            order=interpolation_order,
            cval=np.nan,
        ).data
        for sim, param in zip(sims, params)
    ]
    field_ims_t = np.array(field_ims_t)

    # get blending weights
    if fusion_requires_blending_weights:
        field_ws_t = [
            weights.get_blending_weights(
                target_bb=output_properties,
                source_bb=full_view_bbs[iview],
                affine=params[iview],
                blending_widths=blending_widths,
            )
            for iview in range(len(sims))
        ]
        field_ws_t = field_ws_t * ~np.isnan(field_ims_t)
        field_ws_t = weights.normalize_weights(field_ws_t)
    else:
        field_ws_t = None

    fusion_method_kwargs["transformed_views"] = field_ims_t
    if has_keyword(fusion_func, "params"):
        fusion_method_kwargs["params"] = params
    if fusion_requires_blending_weights:
        fusion_method_kwargs["blending_weights"] = field_ws_t

    # calculate fusion weights if required
    if weights_func is not None and has_keyword(fusion_func, "fusion_weights"):
        weights_func_kwargs["transformed_views"] = field_ims_t
        if has_keyword(weights_func, "params"):
            weights_func_kwargs["params"] = params
        if fusion_requires_blending_weights:
            weights_func_kwargs["blending_weights"] = field_ws_t

        fusion_weights = weights_func(**weights_func_kwargs)
        fusion_method_kwargs["fusion_weights"] = fusion_weights

    fused = func_ignore_nan_warning(
        fusion_func,
        **fusion_method_kwargs,
    )

    # trim overlap
    if trim_overlap_in_pixels > 0:
        fused = fused[
            (slice(trim_overlap_in_pixels, -trim_overlap_in_pixels),) * ndim
        ]

    fused = np.nan_to_num(fused).astype(input_dtype)

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
    combined_stack_props["shape"] = np.max(
        [
            np.ceil(
                (
                    sp["origin"]
                    + sp["shape"] * sp["spacing"]
                    - combined_stack_props["origin"]
                )
                / combined_stack_props["spacing"]
            )
            for sp in stack_props_list
        ],
        axis=0,
    ).astype(np.uint64)

    return combined_stack_props


def get_transformed_stack_vertices(
    stack_keypoints, stack_properties_list, params
):
    ndim = len(stack_properties_list[0]["spacing"])
    vertices = np.zeros(
        (len(stack_properties_list), len(stack_keypoints), ndim)
    )
    for iim, sp in enumerate(stack_properties_list):
        tmp_vertices = stack_keypoints * np.array(sp["shape"]) * np.array(
            sp["spacing"]
        ) + np.array(sp["origin"])
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
    shape = np.ceil((volume[1] - volume[0]) / spacing).astype(np.uint64)

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


def prepare_block_fusion(
    output_zarr_url: str,
    fuse_kwargs: dict,
    zarr_array_creation_kwargs: dict = None,
):
    """
    Prepare chunkwise fusion function and number of blocks
    for embarrassingly parallel fusion
    """

    sims = fuse_kwargs.get("sims")

    output_stack_properties = process_output_stack_properties(
        sims=sims,
        output_stack_properties=fuse_kwargs.pop("output_stack_properties", None),
        output_spacing=fuse_kwargs.pop("output_spacing", None),
        output_origin=fuse_kwargs.pop("output_origin", None),
        output_shape=fuse_kwargs.pop("output_shape", None),
        output_stack_mode=fuse_kwargs.pop("output_stack_mode", "union"),
        transform_key=fuse_kwargs.get("transform_key", None)
    )

    output_chunksize = process_output_chunksize(
        sims, fuse_kwargs.get("output_chunksize", None)
    )

    dims = sims[0].dims
    nsdims = si_utils.get_nonspatial_dims_from_sim(sims[0])
    sdims = si_utils.get_spatial_dims_from_sim(sims[0])
    ns_shape = {dim: len(sims[0].coords[dim]) for dim in nsdims}

    full_output_shape = [ns_shape[dim] for dim in nsdims]\
         + [output_stack_properties['shape'][dim] for dim in sdims]
    full_output_chunksize = [1,] * len(nsdims)\
         + [int(output_chunksize[dim]) for dim in sdims]
    
    normalized_chunks = normalize_chunks(
        shape=full_output_shape,
        chunks=full_output_chunksize)
    
    print(f"Fusing into a an output stack:")
    print("- shape: ", {dim: int(output_stack_properties['shape'][dim])
        if dim in sdims else 1 for dim in dims})
    print("- spacing: ", {k: float(v)
        for k, v in output_stack_properties['spacing'].items()})
    print("- origin: ", {k: float(v)
        for k, v in output_stack_properties['origin'].items()})
    # print(f"- chunksize: {fuse_kwargs.get('output_chunksize', None)}")

    # Create the Zarr array store on disk
    output_zarr_array = zarr.create(
        shape=[int(i) for i in full_output_shape],
        chunks=[int(i) for i in full_output_chunksize],
        dtype=sims[0].data.dtype,
        store=output_zarr_url,  # The path to the directory where the store will be created
        overwrite=True,      # Allows overwriting if the path exists
        **zarr_array_creation_kwargs
        if zarr_array_creation_kwargs is not None else {},
    )

    def fuse_chunk(
            block_id,
            osp=output_stack_properties,
            ns_shape=ns_shape,
            nsdims=nsdims,
            fuse_kwargs=fuse_kwargs,
            output_chunksize=output_chunksize,
            output_zarr_array=output_zarr_array,
            ):
        """
        Fuse a single chunk and write to zarr array.
        """

        sdims = list(osp['shape'].keys())

        normalized_chunks = normalize_chunks(
            shape=[ns_shape[dim] for dim in nsdims] + [osp['shape'][dim] for dim in sdims],
            chunks=(1,) * len(nsdims) +tuple(output_chunksize[dim] for dim in sdims))

        ns_coord = {dim: block_id[idim] for idim, dim in enumerate(nsdims)}

        spatial_chunk_ind = block_id[len(nsdims):]

        chunk_offset = {sdims[idim]: int(np.sum(normalized_chunks[len(nsdims) + idim][:b])) if b > 0 else 0
                        for idim, b in enumerate(spatial_chunk_ind)}
        chunk_offset_phys = {dim: chunk_offset[dim] * osp['spacing'][dim] + osp['origin'][dim]
                            for idim, dim in enumerate(sdims)}
        chunk_shape = {sdims[idim]: normalized_chunks[len(nsdims) + idim][b]
                    for idim, b in enumerate(spatial_chunk_ind)}

        fused = fuse(
            sims=fuse_kwargs.get("sims"),
            **{k: v for k, v in fuse_kwargs.items() if k != "sims"},
            output_origin={dim: chunk_offset_phys[dim] for dim in sdims},
            output_shape={dim: chunk_shape[dim] for dim in sdims},
            output_spacing={dim: osp['spacing'][dim] for dim in sdims},
            ).data

        fused = fused[tuple(slice(ns_coord[dim], ns_coord[dim] + 1) for dim in nsdims)]

        with dask_config.set(scheduler='single-threaded'):
            da.to_zarr(
                fused, output_zarr_array,
                region=tuple(
                    [slice(ns_coord[dim], ns_coord[dim] + 1) for dim in nsdims] +
                    [slice(chunk_offset[dim], chunk_offset[dim] + chunk_shape[dim]) for dim in sdims]),
            )

        return
    
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
