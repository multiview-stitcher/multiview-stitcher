import copy

import numpy as np
import spatial_image as si
import xarray as xr

from multiview_stitcher import param_utils

SPATIAL_DIMS = ["z", "y", "x"]

DEFAULT_SPATIAL_CHUNKSIZES_3D = {"z": 256, "y": 256, "x": 256}
DEFAULT_SPATIAL_CHUNKSIZES_2D = {"y": 1024, "x": 1024}


def get_default_spatial_chunksizes(ndim):
    assert ndim in [2, 3]
    if ndim == 2:
        return DEFAULT_SPATIAL_CHUNKSIZES_2D
    elif ndim == 3:
        return DEFAULT_SPATIAL_CHUNKSIZES_3D


def get_sim_from_array(
    array,
    dims=None,
    scale=None,
    translation=None,
    affine=True,
    transform_key=None,
):
    """
    highest scale sim from msim with affine transforms
    """

    sim = si.to_spatial_image(
        array,
        dims=dims,
        scale=scale,
        translation=translation,
    )

    ndim = get_ndim_from_sim(sim)

    if affine is None:
        affine = np.eye(ndim + 1)

    if transform_key is None:
        transform_key = "affine_manual"

    set_sim_affine(sim, affine, transform_key)

    return sim


def get_spatial_dims_from_sim(sim):
    return [dim for dim in ["z", "y", "x"] if dim in sim.dims]


def get_nonspatial_dims_from_sim(sim):
    sdims = get_spatial_dims_from_sim(sim)
    return [dim for dim in sim.dims if dim not in sdims]


def get_origin_from_sim(sim, asarray=False):
    spatial_dims = get_spatial_dims_from_sim(sim)
    origin = {dim: float(sim.coords[dim][0]) for dim in spatial_dims}

    if asarray:
        origin = np.array([origin[sd] for sd in spatial_dims])

    return origin


def get_shape_from_sim(sim, asarray=False):
    spatial_dims = get_spatial_dims_from_sim(sim)
    shape = {dim: len(sim.coords[dim]) for dim in spatial_dims}

    if asarray:
        shape = np.array([shape[sd] for sd in spatial_dims])

    return shape


def get_spacing_from_sim(sim, asarray=False):
    spatial_dims = get_spatial_dims_from_sim(sim)
    spacing = {
        dim: float(sim.coords[dim][1] - sim.coords[dim][0])
        if len(sim.coords[dim]) > 1
        else 1.0
        for dim in spatial_dims
    }

    if asarray:
        spacing = np.array([spacing[sd] for sd in spatial_dims])

    return spacing


def get_stack_properties_from_sim(sim, transform_key=None, asarray=False):
    stack_properties = {
        "shape": get_shape_from_sim(sim, asarray=asarray),
        "spacing": get_spacing_from_sim(sim, asarray=asarray),
        "origin": get_origin_from_sim(sim, asarray=asarray),
    }

    if transform_key is not None:
        stack_properties["transform"] = get_affine_from_sim(sim, transform_key)

    return stack_properties


def ensure_dim(sim, dim):
    if dim in sim.dims:
        return sim
    else:
        sim = sim.expand_dims([dim], axis=0)

    sim = get_sim_from_xim(sim)

    sim.attrs.update(copy.deepcopy(sim.attrs))

    return sim


def get_sim_from_xim(xim):
    spacing = get_spacing_from_sim(xim)
    origin = get_origin_from_sim(xim)

    sim = si.to_spatial_image(
        xim,
        dims=xim.dims,
        scale=spacing,
        translation=origin,
        t_coords=xim.coords["t"] if "t" in xim.dims else None,
        c_coords=xim.coords["c"] if "c" in xim.dims else None,
    )

    sim.attrs.update(copy.deepcopy(xim.attrs))

    return sim


def get_ndim_from_sim(sim):
    return len(get_spatial_dims_from_sim(sim))


def get_affine_from_sim(sim, transform_key):
    if transform_key not in sim.attrs["transforms"]:
        raise (Exception("Transform key %s not found in sim" % transform_key))

    affine = sim.attrs["transforms"][
        transform_key
    ]  # .reshape((ndim + 1, ndim + 1))

    return affine


def get_tranform_keys_from_sim(sim):
    return list(sim.attrs["transforms"].keys())


def set_sim_affine(sim, xaffine, transform_key, base_transform_key=None):
    if "transforms" not in sim.attrs:
        sim.attrs["transforms"] = {}

    if base_transform_key is not None:
        xaffine = param_utils.rebase_affine(
            xaffine, get_affine_from_sim(sim, transform_key=base_transform_key)
        )

    sim.attrs["transforms"][transform_key] = xaffine

    return


def get_center_of_sim(sim, transform_key=None):
    ndim = get_ndim_from_sim(sim)

    get_spacing_from_sim(sim, asarray=True)
    get_origin_from_sim(sim, asarray=True)

    center = np.array(
        [
            sim.coords[dim][len(sim.coords[dim]) // 2]
            for dim in get_spatial_dims_from_sim(sim)
        ]
    )

    # center = center * spacing + origin

    if transform_key is not None:
        affine = get_affine_from_sim(sim, transform_key=transform_key)
        # select params of first time point if applicable
        sel_dict = {
            dim: affine.coords[dim][0].values
            for dim in affine.dims
            if dim not in ["x_in", "x_out"]
        }
        affine = affine.sel(sel_dict)
        affine = np.array(affine)
        center = np.concatenate([center, np.ones(1)])
        center = np.matmul(affine, center)[:ndim]

    return center


def sim_sel_coords(sim, sel_dict):
    """
    Select coords from sim and its transform attributes
    """

    ssim = sim.copy(deep=True)
    ssim = ssim.sel(sel_dict)

    # sel transforms which are xr.Datasets in the msim attributes
    for data_var in sim.attrs["transforms"]:
        for k, v in sel_dict.items():
            if k in sim.attrs["transforms"][data_var].dims:
                ssim.attrs["transforms"][data_var] = ssim.attrs["transforms"][
                    data_var
                ].sel({k: v})

    return ssim


def get_sim_field(sim, ns_coords=None):
    sdims = get_spatial_dims_from_sim(sim)
    nsdims = [dim for dim in sim.dims if dim not in sdims]

    if not len(nsdims):
        return sim

    if ns_coords is None:
        ns_coords = {dim: sim.coords[dim][0] for dim in nsdims}

    sim_field = sim_sel_coords(sim, ns_coords)

    return sim_field


def process_fields(sim, func, **func_kwargs):
    sdims = get_spatial_dims_from_sim(sim)

    return xr.apply_ufunc(
        func,
        sim,
        input_core_dims=[sdims],
        output_core_dims=[sdims],
        vectorize=True,
        dask="allowed",
        keep_attrs=True,
        kwargs=func_kwargs,
    )
