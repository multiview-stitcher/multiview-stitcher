from functools import wraps
from pathlib import Path

import datatree
import multiscale_spatial_image as msi
import spatial_image as si
import xarray as xr

from multiview_stitcher import spatial_image_utils


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


def get_transform_from_msim(msim, transform_key=None):
    """
    Get transform from msim. If transform_key is None, get the transform from the first scale.
    """

    return msim["scale0"][transform_key]


# def multiscale_sel_coords(msim, sel_dict):

#     out_msim = msim.copy(deep=True)
#     for child in msim.children.keys():
#         for data_var in msim[child].data_vars:
#             tmpsel = {dim: sel_vals for dim, sel_vals in sel_dict.items()
#                      if dim in out_msim[child][data_var].dims}
#             import pdb; pdb.set_trace()
#             out_msim[child][data_var] = \
#                 out_msim[child][data_var].sel(tmpsel, drop=False)
#             out_msim[child][data_var] = \
#                 out_msim[child][data_var].assign_coords(tmpsel)
#             # print(tmpsel)

#             # for dim, sel_vals in tmpsel.items():
#             #     out_msim[child][data_var].coords[dim] = sel_vals

#     return out_msim


def multiscale_sel_coords(msim, sel_dict):
    """ """

    # Somehow .sel on a datatree does not work when
    # attributes are present. So we remove them and
    # add them back after sel.

    attrs = msim.attrs.copy()
    msim.attrs = {}
    msim = msim.sel(sel_dict)
    msim.attrs = attrs

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


def multiscale_spatial_image_from_zarr(path):
    ndim = spatial_image_utils.get_ndim_from_sim(
        datatree.open_datatree(path, engine="zarr")["scale0/image"]
    )

    if ndim == 2:
        chunks = {"y": 256, "x": 256}
    elif ndim == 3:
        # chunks = {'z': 64, 'y': 64, 'x': 64}
        chunks = {"z": 256, "y": 256, "x": 256}

    multiscale = datatree.open_datatree(path, engine="zarr", chunks=chunks)

    # compute transforms
    sorted_scales = get_sorted_scale_keys(multiscale)
    for scale in sorted_scales:
        for data_var in multiscale[scale].data_vars:
            if data_var == "image":
                continue
            multiscale[scale][data_var] = multiscale[scale][data_var].compute()

    return multiscale


def get_optimal_multi_scale_factors_from_sim(sim, min_size=512):
    """
    This is currently simply downscaling z and xy until a minimum size is reached.
    Probably it'd make more sense to downscale considering the dims spacing.
    """

    spatial_dims = spatial_image_utils.get_spatial_dims_from_sim(sim)
    current_shape = {dim: len(sim.coords[dim]) for dim in spatial_dims}
    factors = []
    while 1:
        curr_factors = {
            dim: 2 if current_shape[dim] >= min_size else 1
            for dim in current_shape
        }
        if max(curr_factors.values()) == 1:
            break
        current_shape = {
            dim: int(current_shape[dim] / curr_factors[dim])
            for dim in current_shape
        }
        factors.append(curr_factors)

    return factors


def get_transforms_from_dataset_as_dict(dataset):
    transforms_dict = {}
    for data_var, transform in dataset.items():
        if data_var == "image":
            continue
        transform_key = data_var
        transforms_dict[transform_key] = transform
    return transforms_dict


def get_sim_from_msim(msim, scale="scale0"):
    """
    highest scale sim from msim with affine transforms
    """
    sim = msim["%s/image" % scale].copy()
    sim.attrs["transforms"] = get_transforms_from_dataset_as_dict(
        msim["scale0"]
    )

    return sim


def get_msim_from_sim(sim, scale_factors=None):
    """
    highest scale sim from msim with affine transforms
    """

    spacing = spatial_image_utils.get_spacing_from_sim(sim)
    origin = spatial_image_utils.get_origin_from_sim(sim)

    if "c" in sim.dims and "t" in sim.dims:
        sim = sim.transpose(
            *tuple(
                ["t", "c"] + [dim for dim in sim.dims if dim not in ["c", "t"]]
            )
        )
        c_coords = sim.coords["c"].values
    else:
        c_coords = None

    sim_attrs = sim.attrs.copy()

    # view_sim.name = str(view)
    sim = si.to_spatial_image(
        sim.data,
        dims=sim.dims,
        c_coords=c_coords,
        scale=spacing,
        translation=origin,
        t_coords=sim.coords["t"].values,
    )

    if scale_factors is None:
        scale_factors = get_optimal_multi_scale_factors_from_sim(sim)

    chunks = {dim: 256 if dim not in ["c", "t"] else 1 for dim in sim.dims}

    msim = msi.to_multiscale(
        sim,
        chunks=chunks,
        scale_factors=scale_factors,
    )

    if "transforms" in sim_attrs:
        scale_keys = get_sorted_scale_keys(msim)
        for sk in scale_keys:
            for transform_key, transform in sim_attrs["transforms"].items():
                msim[sk][transform_key] = transform

    return msim


def set_affine_transform(
    msim, xaffine, transform_key, base_transform_key=None
):
    if not isinstance(xaffine, xr.DataArray):
        xaffine = xr.DataArray(xaffine, dims=["t", "x_in", "x_out"])

    if base_transform_key is not None:
        xaffine = spatial_image_utils.matmul_xparams(
            xaffine,
            get_transform_from_msim(msim, transform_key=base_transform_key),
        )

    scale_keys = get_sorted_scale_keys(msim)
    for sk in scale_keys:
        msim[sk][transform_key] = xaffine


def ensure_time_dim(msim):

    if "t" in msim["scale0/image"].dims:
        return msim

    scale_keys = get_sorted_scale_keys(msim)
    for sk in scale_keys:
        for data_var in msim[sk].data_vars:

            if data_var == "image":
                msim[sk][data_var] = spatial_image_utils.ensure_time_dim(
                    msim[sk][data_var])
            else:

                if "t" in msim[sk][data_var].dims:
                    continue
                else:
                    msim[sk][data_var] =\
                         msim[sk][data_var].expand_dims(["t"], axis=0)

    return msim


def get_first_scale_above_target_spacing(msim, target_spacing, dim="y"):
    sorted_scale_keys = get_sorted_scale_keys(msim)

    for scale in sorted_scale_keys:
        scale_spacing = spatial_image_utils.get_spacing_from_sim(
            msim[scale]["image"]
        )[dim]
        if scale_spacing > target_spacing:
            break

    return scale


def get_ndim(msim):
    return spatial_image_utils.get_ndim_from_sim(get_sim_from_msim(msim))


def get_spatial_dims(msim):
    return spatial_image_utils.get_spatial_dims_from_sim(
        get_sim_from_msim(msim)
    )


def get_dims(msim):
    return get_sim_from_msim(msim).dims
