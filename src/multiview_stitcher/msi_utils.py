import copy
import os
from functools import wraps
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr
from xarray import DataTree

from multiview_stitcher import param_utils
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher import zarr_utils


def is_msim(image):
    """Return whether ``image`` follows the multiscale DataTree layout."""
    return isinstance(image, DataTree)


def _mean_dtype(array, axis=None):
    return array.mean(axis=axis).astype(array.dtype)


def _chunk_sim(sim, chunks):
    return sim.chunk({dim: chunks[dim] for dim in sim.dims if dim in chunks})


def _sim_to_dataset(sim):
    # Strip attrs (transforms are re-attached as data_vars at the msim level).
    # Chunk hints live in xarray encoding, which survives to_dataset / DataTree,
    # so no dim/chunk bookkeeping needs to be carried on attrs.
    dataset = sim.to_dataset(name="image")
    dataset.attrs = {}
    dataset["image"].attrs = {}
    return dataset


def _downsample_sim(sim, scale_factor, chunks):
    spatial_dims = si_utils.get_spatial_dims_from_sim(sim)
    if not isinstance(scale_factor, dict):
        scale_factor = {dim: int(scale_factor) for dim in spatial_dims}

    scale_factor = {
        dim: int(scale_factor.get(dim, 1)) for dim in spatial_dims
    }

    downsampled = sim.coarsen(
        {dim: scale_factor[dim] for dim in spatial_dims},
        boundary="trim",
    ).reduce(_mean_dtype)

    spacing = si_utils.get_spacing_from_sim(sim)
    origin = si_utils.get_origin_from_sim(sim)

    downsampled_sim = si_utils.to_spatial_image(
        downsampled.data,
        dims=sim.dims,
        scale={dim: spacing[dim] * scale_factor[dim] for dim in spatial_dims},
        translation={
            dim: origin[dim] + (scale_factor[dim] - 1) * spacing[dim] / 2
            for dim in spatial_dims
        },
        t_coords=sim.coords["t"].values if "t" in sim.coords else None,
        c_coords=sim.coords["c"].values if "c" in sim.coords else None,
    )
    downsampled_sim.attrs.update(copy.deepcopy(sim.attrs))

    return _chunk_sim(downsampled_sim, chunks)


def get_store_decorator(store_path, store_overwrite=False):
    """
    Generator of decorators meant for functions that read some file (non lazy?) into a msi.
    Decorators stores resulting msi in a zarr and returns a new msi loaded from the store.
    """
    if store_path is None:
        return lambda func: func

    def store_decorator(func):
        """
        store_decorator takes care of caching msi on disk
        """

        @wraps(func)
        def wrapper_decorator(*args, **kwargs):
            if not store_path.exists() or store_overwrite:
                msi = func(*args, **kwargs)
                msi.to_zarr(Path(store_path))

            return multiscale_spatial_image_from_zarr(Path(store_path))

        return wrapper_decorator

    return store_decorator


def get_transform_from_msim(msim, transform_key):
    """
    Get transform from msim. If transform_key is None, get the transform from the first scale.
    """

    return msim["scale0"][transform_key]


def multiscale_sel_coords(msim, sel_dict):
    """ """

    # Somehow .sel on a datatree does not work when
    # attributes are present. So we remove them and
    # add them back after sel.

    attrs = msim.attrs.copy()
    msim.attrs = {}
    msim = msim.sel(sel_dict)
    msim.attrs = attrs

    if "point_sets" in list(msim.keys()):
        for points_key in list(msim["point_sets"].keys()):
            point_set = msim[f"point_sets/{points_key}"].to_dataset()
            msim[f"point_sets/{points_key}"] = si_utils.point_set_sel_coords(
                point_set,
                sel_dict,
            )

    return msim  # .sel(sel_dict)


def get_sorted_scale_keys(msim):
    sorted_scale_keys = [
        "scale%s" % scale
        for scale in sorted(
            [
                int(scale_key.split("scale")[-1])
                for scale_key in list(msim.keys())
                if "scale" in scale_key
            ]
        )
    ]  # there could be transforms also

    return sorted_scale_keys


def multiscale_spatial_image_from_zarr(path, chunks=None):
    dims = xr.open_datatree(path, engine="zarr")["scale0/image"].dims
    sdims = [dim for dim in dims if dim in si_utils.SPATIAL_DIMS]
    ndim = len(sdims)
    nsdims = [dim for dim in dims if dim not in sdims]

    if chunks is None:
        chunks = si_utils.get_default_spatial_chunksizes(ndim)

        for nsdim in nsdims:
            chunks[nsdim] = 1

    multiscale = xr.open_datatree(path, engine="zarr", chunks=chunks)

    # compute transforms
    sorted_scales = get_sorted_scale_keys(multiscale)
    for scale in sorted_scales:
        for data_var in multiscale[scale].data_vars:
            if data_var == "image":
                continue
            multiscale[scale][data_var] = multiscale[scale][data_var].compute()

    return multiscale


def multiscale_spatial_image_to_zarr(msim, path):
    """
    This is a workaround for a bug in xarray/zarr:
    https://stackoverflow.com/questions/67476513/zarr-not-respecting-chunk-size-from-xarray-and-reverting-to-original-chunk-size
    """
    for scale_key in get_sorted_scale_keys(msim):
        if "chunks" in msim[scale_key]["image"].encoding:
            del msim[scale_key]["image"].encoding["chunks"]
    msim.to_zarr(path)


def update_msim_transforms_zarr(msim, path, overwrite=False):
    """
    Update (or write) multiview-stitcher flavor transform metadata to zarr store.
    Concerns all data variables except "image".
    """

    path = Path(path)

    if not path.exists():
        raise ValueError("Path does not exist")

    msim_disk = multiscale_spatial_image_from_zarr(path)
    msim_disk_scale_keys = get_sorted_scale_keys(msim_disk)
    for scale_key in get_sorted_scale_keys(msim):
        for data_var in [
            dv for dv in msim[scale_key].data_vars if dv != "image"
        ]:
            if (
                scale_key in msim_disk_scale_keys
                and data_var in msim_disk[scale_key].data_vars
                and not overwrite
            ):
                continue
            msim[scale_key][data_var].to_zarr(
                path,
                group=os.path.join(scale_key, data_var),
                mode="w",
            )

    return


def calc_resolution_levels(
    spatial_shape,
    downscale_factors_per_spatial_dim=None,
    min_shape=100
):
    """
    Calculate resolution levels given spatial shape and downscale factors per spatial dimension.
    Returns list of spatial shapes, relative downscale factors and absolute downscale factors.

    Note that the highest resolution level (input spatial shape) is included in the output lists.
    """
    sdims = list(spatial_shape.keys())

    if downscale_factors_per_spatial_dim is None:
        downscale_factors_per_spatial_dim = {dim: 2 for dim in sdims}

    res_shapes = [spatial_shape]
    res_rel_factors = [{dim: 1 for dim in sdims}]
    res_abs_factors = [{dim: 1 for dim in sdims}]
    while True:
        new_rel_factors = {
            dim: downscale_factors_per_spatial_dim[dim]
            if res_shapes[-1][dim] // downscale_factors_per_spatial_dim[dim]
            > min_shape
            else 1
            for dim in sdims
        }

        new_abs_factors = {
            dim: res_abs_factors[-1][dim] * new_rel_factors[dim]
            for dim in sdims
        }
        new_shape = {
            dim: res_shapes[-1][dim] // new_rel_factors[dim] for dim in sdims
        }
        if not any(new_rel_factors[dim] > 1 for dim in sdims):
            break

        res_shapes.append(new_shape)
        res_rel_factors.append(
            new_rel_factors
        )
        res_abs_factors.append(
            new_abs_factors
        )

    return res_shapes, res_rel_factors, res_abs_factors


def get_transforms_from_dataset_as_dict(dataset):
    transforms_dict = {}
    for data_var, transform in dataset.items():
        if data_var == "image":
            continue
        transform_key = data_var
        transforms_dict[transform_key] = transform
    return transforms_dict


def _get_inherited_coords_for_sim(msim, scale, sim):
    inherited_coords = {}
    for coord_source in (msim, msim[scale]):
        for coord_name, coord in coord_source.coords.items():
            if coord_name in sim.coords:
                continue
            if not all(dim in sim.dims for dim in coord.dims):
                continue
            inherited_coords[coord_name] = coord

    return inherited_coords


def get_sim_from_msim(msim, scale="scale0"):
    """
    highest scale sim from msim with affine transforms
    """
    sim = msim["%s/image" % scale].copy()
    inherited_coords = _get_inherited_coords_for_sim(msim, scale, sim)
    if inherited_coords:
        sim = sim.assign_coords(inherited_coords)
    sim.attrs["transforms"] = get_transforms_from_dataset_as_dict(
        msim["scale0"]
    )
    if "point_sets" in list(msim.keys()):
        for points_key in list(msim["point_sets"].keys()):
            si_utils.set_point_set(
                sim,
                get_point_set(msim, points_key=points_key),
                points_key=points_key,
            )

    return sim


def get_msim_from_sim(sim, scale_factors=None, chunks=None):
    """
    highest scale sim from msim with affine transforms
    """

    ndim = si_utils.get_ndim_from_sim(sim)

    sim = sim.copy()
    sim_attrs = copy.deepcopy(sim.attrs)
        
    spatial_shape = si_utils.get_shape_from_sim(sim)

    if scale_factors is None:
        scale_factors = calc_resolution_levels(spatial_shape)[1]
        scale_factors = scale_factors[1:]  # remove input scale

    if chunks is None and len(scale_factors) > 0:
        sim_backend = getattr(sim.variable, "_data", None)
        if isinstance(sim_backend, da.Array):
            chunks = {
                dim: sim_backend.chunksize[idim]
                for idim, dim in enumerate(sim.dims)
            }
        else:
            spatial_chunksizes = si_utils.get_default_spatial_chunksizes(ndim)
            chunks = {
                dim: spatial_chunksizes[dim] if dim not in ["c", "t"] else 1
                for dim in sim.dims
            }

    if si_utils.is_xarray_zarr_backed(sim):
        sims = [
            sim if len(scale_factors) == 0 else si_utils.ensure_dask_backed_dataarray(sim)
        ]
    else:
        sims = [sim]

    for scale_factor in scale_factors:
        sims.append(_downsample_sim(sims[-1], scale_factor, chunks))

    msim = DataTree.from_dict(
        {
            f"scale{iscale}": _sim_to_dataset(curr_sim)
            for iscale, curr_sim in enumerate(sims)
        }
    )

    if "transforms" in sim_attrs:
        scale_keys = get_sorted_scale_keys(msim)
        for sk in scale_keys:
            for transform_key, transform in sim_attrs["transforms"].items():
                msim[sk][transform_key] = transform

    if "point_sets" in sim_attrs:
        for points_key, point_set in sim_attrs["point_sets"].items():
            set_point_set(msim, point_set, points_key=points_key)

    return msim


def get_msim_from_sims(sims):
    """
    Build a multiscale image from already-computed resolution levels.

    Input sims are ordered by decreasing spatial shape. Transform metadata is
    taken from the highest-resolution sim and copied to every output scale.
    """
    # Accept any iterable, then work with a local list so ordering is explicit.
    sims = list(sims)
    if not sims:
        raise ValueError("sims must contain at least one image.")

    # All levels must describe the same axes to form a valid pyramid.
    dims = sims[0].dims
    for sim in sims[1:]:
        if sim.dims != dims:
            raise ValueError("All sims must have the same dimensions.")

    sdims = si_utils.get_spatial_dims_from_sim(sims[0])

    def shape_key(sim):
        shape = si_utils.get_shape_from_sim(sim)
        return tuple(shape[dim] for dim in sdims)

    # Put the highest-resolution image at scale0 regardless of caller order.
    sims = sorted(sims, key=shape_key, reverse=True)
    shapes = [si_utils.get_shape_from_sim(sim) for sim in sims]
    for prev_shape, shape in zip(shapes, shapes[1:]):
        if not all(shape[dim] <= prev_shape[dim] for dim in sdims):
            # Incomparable anisotropic shapes cannot be ordered into pyramid levels.
            raise ValueError(
                "Spatial shapes must decrease monotonically across "
                "resolution levels."
            )

    # Copy sims before normalizing transform attrs so caller objects are unchanged.
    sims = [sim.copy() for sim in sims]
    transform_keys = (
        si_utils.get_tranform_keys_from_sim(sims[0])
        if "transforms" in sims[0].attrs
        else []
    )

    # Lower-resolution sims inherit exactly the transform keys from scale0.
    for sim in sims[1:]:
        sim.attrs["transforms"] = {}
        for transform_key in transform_keys:
            si_utils.set_sim_affine(
                sim,
                si_utils.get_affine_from_sim(sims[0], transform_key),
                transform_key,
            )

    msim_dict = {}
    for iscale, sim in enumerate(sims):
        curr_msim = get_msim_from_sim(sim, scale_factors=[])
        msim_dict[f"scale{iscale}"] = curr_msim["scale0"]

    msim = DataTree.from_dict(msim_dict)
    # Re-apply through the msim helper to keep transform storage consistent.
    for transform_key in transform_keys:
        set_affine_transform(
            msim,
            si_utils.get_affine_from_sim(sims[0], transform_key),
            transform_key,
        )

    return msim


def msim_map_blocks(msim, func, *args, **kwargs):
    """
    Apply ``dask.array.Array.map_blocks`` to the image data in every scale.

    This keeps coordinates and non-image data variables intact while lazily
    transforming each image chunk. For example,
    ``msim_map_blocks(msim, cupy.asarray)`` makes computed chunks CuPy-backed.
    Additional positional and keyword arguments are forwarded to
    ``dask.array.Array.map_blocks``.
    """

    def _map_dataset(ds):
        if "image" not in ds.data_vars:
            return ds

        return ds.assign(
            image=ds.image.copy(
                data=ds.image.data.map_blocks(func, *args, **kwargs)
            )
        )

    return msim.map_over_datasets(_map_dataset)


def set_point_set(msim, points, points_key="beads"):
    """
    Attach a named point set to a multiscale spatial image.

    Point positions are expected in intrinsic physical coordinates, i.e. image
    origin and spacing have already been applied.
    ``points`` may be an array with shape ``(n_points, n_spatial_dims)``;
    columns must follow the image spatial dimension order.

    Examples
    --------
    Set points from a plain NumPy array. Columns follow the image spatial
    dimension order:

    >>> points = np.array([[10.0, 20.0], [12.0, 24.0]])
    >>> set_point_set(msim, points, points_key="beads")

    Set points from NumPy values with explicit dimension labels:

    >>> dim_names = ["y", "x"]
    >>> point_set = xr.DataArray(
    ...     points,
    ...     dims=["point_id", "dim"],
    ...     coords={"dim": dim_names},
    ...     name="position",
    ... ).to_dataset()
    >>> set_point_set(msim, point_set, points_key="beads")
    """

    point_set = si_utils._coerce_point_set(
        points,
        sdims=get_spatial_dims(msim),
        nscoords=si_utils._get_nscoords_from_sim(msim["scale0/image"]),
    )
    msim[f"point_sets/{points_key}"] = point_set

    return


def get_point_set(msim, points_key="beads"):
    """
    Get a named point set from a multiscale spatial image.

    Examples
    --------
    Retrieve the stored points, their axes order, and spatial dimension
    coordinates:

    >>> position = get_point_set(msim, points_key="beads")["position"]
    >>> position.dims
    ('t', 'c', 'point_id', 'dim')
    >>> list(position.coords["dim"].values)
    ['y', 'x']
    >>> position.isel(t=0, c=0).values
    array([[10., 20.],
           [12., 24.]])
    """

    if (
        "point_sets" not in list(msim.keys())
        or points_key not in list(msim["point_sets"].keys())
    ):
        raise KeyError(f"Point set {points_key!r} not found in msim.")

    return si_utils._coerce_point_set(
        msim[f"point_sets/{points_key}"].to_dataset()
    )


def set_affine_transform(
    msim, xaffine=None, transform_key=None, base_transform_key=None
):
    if transform_key is None:
        raise ValueError("transform_key must be provided")

    if xaffine is None:
        ndim = get_ndim(msim)
        xaffine = param_utils.identity_transform(ndim)

    if not isinstance(xaffine, xr.DataArray):
        xaffine = xr.DataArray(xaffine, dims=["t", "x_in", "x_out"])

    if base_transform_key is not None:
        xaffine = param_utils.rebase_affine(
            xaffine,
            get_transform_from_msim(msim, transform_key=base_transform_key),
        )

    scale_keys = get_sorted_scale_keys(msim)
    for sk in scale_keys:
        msim[sk][transform_key] = xaffine


def ensure_dim(msim, dim):
    if dim in msim["scale0/image"].dims:
        return msim

    scale_keys = get_sorted_scale_keys(msim)
    for sk in scale_keys:
        for data_var in msim[sk].data_vars:
            if data_var == "image":
                msim[sk][data_var] = si_utils.ensure_dim(
                    msim[sk][data_var], dim
                )
            else:
                if dim in msim[sk][data_var].dims:
                    continue
                else:
                    msim[sk][data_var] = msim[sk][data_var].expand_dims(
                        [dim], axis=0
                    )

    return msim


def get_first_scale_above_target_spacing(msim, target_spacing, dim="y"):
    sorted_scale_keys = get_sorted_scale_keys(msim)

    for scale in sorted_scale_keys:
        scale_spacing = si_utils.get_spacing_from_sim(msim[scale]["image"])[
            dim
        ]
        if scale_spacing > target_spacing:
            break

    return scale


def get_res_level_from_spacing(msim, spacing: dict) -> int:
    """
    Find the highest (coarsest) resolution level whose actual spacing is
    still lower or equal to the target spacing in every requested dimension.

    Parameters
    ----------
    msim : MultiscaleSpatialImage
        Multiscale spatial image.
    spacing : dict[str, float]
        Target spacing per spatial dimension.  Only dimensions present in
        this dict are considered; others are ignored.

    Returns
    -------
    int
        Integer index of the selected resolution level (e.g. ``0`` for
        ``"scale0"``, ``1`` for ``"scale1"``, etc.).
    """
    sorted_scale_keys = get_sorted_scale_keys(msim)

    best_level = 0
    for i, scale_key in enumerate(sorted_scale_keys):
        sim = get_sim_from_msim(msim, scale=scale_key)
        actual_spacing = si_utils.get_spacing_from_sim(sim)
        if all(actual_spacing[dim] <= spacing[dim] for dim in spacing):
            best_level = i
        else:
            break

    return best_level


def get_res_level_from_binning_factors(msim, binning_factors):
    """
    Find the optimal resolution level for the given binning factors.
    
    Returns the lowest resolution level such that the downsampling factors 
    for all dimensions are >= the requested binning factors and are integer 
    multiples of the binning factors.
    
    Parameters
    ----------
    msim : MultiscaleSpatialImage
        Multiscale spatial image
    binning_factors : dict[str, int]
        Target binning factors for each spatial dimension
        
    Returns
    -------
    str
        Scale key (e.g., 'scale0', 'scale1', etc.)
    dict[str, int]
        Remaining binning factors to apply after selecting the resolution level
    """
    sorted_scale_keys = get_sorted_scale_keys(msim)
    
    # Get the shape at scale0
    sim0 = get_sim_from_msim(msim, scale="scale0")
    spatial_dims = si_utils.get_spatial_dims_from_sim(sim0)
    shape0 = {dim: len(sim0.coords[dim]) for dim in spatial_dims}
    
    # Start with scale0 as the default
    best_scale = "scale0"
    best_remaining_binning = binning_factors.copy()
    
    for scale_key in sorted_scale_keys:
        sim = get_sim_from_msim(msim, scale=scale_key)
        shape = {dim: len(sim.coords[dim]) for dim in spatial_dims}
        
        # Calculate the downsampling factor for this scale
        scale_factors = {
            dim: shape0[dim] / shape[dim] for dim in spatial_dims
        }
        
        # Check if this scale is suitable for all dimensions
        valid = True
        
        for dim in spatial_dims:
            if dim not in binning_factors:
                continue
                
            target_factor = binning_factors[dim]
            actual_factor = scale_factors[dim]
            
            # Check if actual_factor is close to an integer
            if not np.isclose(actual_factor, round(actual_factor), rtol=1e-6):
                valid = False
                break
            
            actual_factor_int = int(round(actual_factor))
            
            # Scale factor must be <= target (we can't use a lower resolution than requested)
            if actual_factor_int > target_factor:
                valid = False
                break
                
            # Target must be an integer multiple of scale factor
            if target_factor % actual_factor_int != 0:
                valid = False
                break
        
        if valid:
            # This scale is usable, calculate remaining binning needed
            remaining_binning = {}
            for dim in spatial_dims:
                if dim not in binning_factors:
                    remaining_binning[dim] = 1
                else:
                    actual_factor_int = int(round(scale_factors[dim]))
                    # Remaining binning = target / actual
                    remaining_binning[dim] = binning_factors[dim] // actual_factor_int
            
            best_scale = scale_key
            best_remaining_binning = remaining_binning
            # Continue to check if there's an even better (lower resolution) scale
    
    return best_scale, best_remaining_binning


def get_ndim(msim):
    return si_utils.get_ndim_from_sim(get_sim_from_msim(msim))


def get_spatial_dims(msim):
    return si_utils.get_spatial_dims_from_sim(get_sim_from_msim(msim))


def get_dims(msim):
    return get_sim_from_msim(msim).dims


def correct_multiscale_origins(msim):
    """
    Correct origins of all scales in msim to match the origin of the first scale,
    so that transforms into the intrinsic coordinate system (as defined by
    OME-Zarr v0.6) are correct.
    See also https://github.com/ome/ngff-spec/pull/125
    """

    sks = get_sorted_scale_keys(msim)
    spacing0 = si_utils.get_spacing_from_sim(get_sim_from_msim(msim, sks[0]))
    origin0 = si_utils.get_origin_from_sim(get_sim_from_msim(msim, sks[0]))
    sdims = get_spatial_dims(msim)

    sim0 = get_sim_from_msim(msim, sks[0])
    shape0 = {dim: len(sim0.coords[dim]) for dim in sdims}
    msim = msim.map_over_datasets(lambda ds: xr.Dataset(
        {'image': ds.image.assign_coords(
            {dim: ds.image.coords[dim] - ds.image.coords[dim].values[0] + origin0[dim]\
              + (np.round(shape0[dim] / len(ds.image.coords[dim])) - 1) / 2 * spacing0[dim]
              for dim in sdims}
                )} | \
        {t: ds.data_vars[t] for t in ds.data_vars if t != 'image'})
        if "image" in ds.data_vars else ds)

    return msim


def _scale_sims_concatable(scale_sims, dim):
    """True when every scale can be concatenated lazily along ``dim`` in zarr."""
    for sims in scale_sims:
        if not (len(sims) > 1 and dim in sims[0].dims):
            return False
        if not si_utils._zarr_backed_combine_applicable(sims):
            return False
        axis = list(sims[0].dims).index(dim)
        zarrays = [si_utils._get_xarray_zarr_array(sim) for sim in sims]
        if not zarr_utils.is_chunk_aligned_concatenate(zarrays, axis):
            return False
    return True


# define a function to combine msims along a given dimension
def concat(msims, concat_kwargs={}, dim='c'):

    msims = list(msims)
    scale_keys = get_sorted_scale_keys(msims[0])

    # Lazy fast path: when every scale is zarr-backed and chunk-aligned along
    # ``dim`` (always so for chunk-size-1 axes such as ``c``/``t``), combine each
    # scale's image via the zarr-aware si_utils.concat so the result stays
    # zarr-backed instead of being materialized by the dataset-level xr.concat.
    # The combined per-scale sims are resolution levels of the combined image,
    # so get_msim_from_sims reassembles them into a DataTree (keeping the lazy
    # zarr backing and propagating the transforms).
    scale_sims = [
        [get_sim_from_msim(msim, scale=sk) for msim in msims]
        for sk in scale_keys
    ]
    if scale_keys and _scale_sims_concatable(scale_sims, dim):
        return get_msim_from_sims(
            [si_utils.concat(sims, dim=dim) for sims in scale_sims]
        )

    with xr.set_options(keep_attrs=True):
        return xr.DataTree.from_dict(
            {sk: xr.concat(
                [msim[sk].dataset for msim in msims],
                dim=dim,
                data_vars='different',
                compat="equals",
                **concat_kwargs)
                for sk in list(msims)[0].keys()}
        )


def stack(msims, dim="t"):
    """Stack msims along a new dimension ``dim`` (default "t").

    Each scale is stacked via the zarr-aware :func:`si_utils.stack`, so a stack
    of zarr-backed msims stays lazily zarr-backed (new axis with chunk size 1).
    The stacked per-scale sims are reassembled with get_msim_from_sims.
    """
    msims = list(msims)
    scale_keys = get_sorted_scale_keys(msims[0])
    scale_sims = [
        [get_sim_from_msim(msim, scale=sk) for msim in msims]
        for sk in scale_keys
    ]
    return get_msim_from_sims(
        [si_utils.stack(sims, dim=dim) for sims in scale_sims]
    )
