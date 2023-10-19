import copy

import numpy as np
import spatial_image as si
import transforms3d as t3d
import xarray as xr

SPATIAL_DIMS = ["z", "y", "x"]


def assign_si_coords_from_params(sim, p=None):
    """
    Assume that matrix p (shape ndim+1) is given with dim order
    equal to those in im (should be Z, Y, X)
    """

    spatial_dims = [dim for dim in ["z", "y", "x"] if dim in sim.dims]
    ndim = len(spatial_dims)

    # if ndim==2 temporarily expand to three dims to use
    # transformations.py for decomposition
    if ndim == 2:
        M = np.eye(4)
        M[1:, 1:] = p
        p = M.copy()

    translate, angles, scale, _ = t3d.affines.decompose(p)
    direction_matrix = angles
    # use t3d.affines.compose here for consistency

    if ndim == 2:
        scale = scale[1:]
        translate = translate[1:]
        direction_matrix = direction_matrix[1:, 1:]

    # assign new coords
    for idim, dim in enumerate(spatial_dims):
        coords = np.linspace(0, len(sim.coords[dim]) - 1, len(sim.coords[dim]))
        coords *= scale[idim]
        coords += translate[idim]
        sim.coords[dim] = coords

    sim.attrs["direction"] = direction_matrix

    return sim


def compose_params(origin, spacing):
    ndim = len(origin)

    if ndim == 2:
        origin = np.concatenate([[0.0], origin])
        spacing = np.concatenate([[1.0], spacing])

    M = t3d.affines.compose(T=origin, R=np.identity(3, np.float64), Z=spacing)

    if ndim == 2:
        M = M[1:, 1:]

    return M


def get_data_to_world_matrix_from_spatial_image(sim):
    spatial_dims = [dim for dim in ["z", "y", "x"] if dim in sim.dims]

    ndim = len(spatial_dims)
    p = np.eye(ndim + 1)

    scale, offset = {}, {}
    for _, dim in enumerate(spatial_dims):
        coords = sim.coords[dim]

        if len(coords) > 1:
            scale[dim] = coords[1] - coords[0]
        else:
            scale[dim] = 1

        offset[dim] = coords[0]

    S = np.diag([scale[dim] for dim in spatial_dims] + [1])
    T = np.eye(ndim + 1)
    T[:ndim, ndim] = [offset[dim] for dim in spatial_dims]

    # direction not implemented (yet?)
    # p = np.matmul(T, np.matmul(S, sim.attrs['direction']))
    p = np.matmul(T, S)

    return p


def get_spatial_dims_from_sim(sim):
    return [dim for dim in ["z", "y", "x"] if dim in sim.dims]


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


def ensure_time_dim(sim):
    if "t" in sim.dims:
        return sim
    else:
        sim = sim.expand_dims(["t"], axis=0)

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


def get_affine_from_sim(sim, transform_key=None):
    if transform_key not in sim.attrs["transforms"]:
        raise (Exception("Transform key %s not found in sim" % transform_key))

    affine = sim.attrs["transforms"][
        transform_key
    ]  # .reshape((ndim + 1, ndim + 1))

    return affine


def get_tranform_keys_from_sim(sim):
    return list(sim.attrs["transforms"].keys())


def set_sim_affine(sim, xaffine, transform_key=None, base_transform_key=None):
    if "transforms" not in sim.attrs:
        sim.attrs["transforms"] = {}

    if base_transform_key is not None:
        xaffine = matmul_xparams(
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


def identity_transform(ndim, t_coords=None):
    if t_coords is None:
        params = xr.DataArray(np.eye(ndim + 1), dims=["x_in", "x_out"])
    else:
        params = xr.DataArray(
            len(t_coords) * [np.eye(ndim + 1)],
            dims=["t", "x_in", "x_out"],
            coords={"t": t_coords},
        )

    return params


def affine_to_xr(affine, t_coords=None):
    if t_coords is None:
        params = xr.DataArray(affine, dims=["x_in", "x_out"])
    else:
        params = xr.DataArray(
            len(t_coords) * [affine],
            dims=["t", "x_in", "x_out"],
            coords={"t": t_coords},
        )

    return params


def matmul_xparams(xparams1, xparams2):
    return xr.apply_ufunc(
        np.matmul,
        xparams1,
        xparams2,
        input_core_dims=[["x_in", "x_out"]] * 2,
        output_core_dims=[["x_in", "x_out"]],
        dask="parallelized",
        vectorize=True,
        join="inner",
    )


def invert_xparams(xparams):
    return xr.apply_ufunc(
        np.linalg.inv,
        xparams,
        input_core_dims=[["x_in", "x_out"]],
        output_core_dims=[["x_in", "x_out"]],
        vectorize=False,
        # dask='allowed',
        dask="parallelized",
    )
