import copy
from typing import Optional, Union

import dask.array as da
import numpy as np
import spatial_image as si
import xarray as xr
from numpy._typing import ArrayLike

from multiview_stitcher import param_utils

DEFAULT_TRANSFORM_KEY = "affine_metadata"

SPATIAL_DIMS = ["z", "y", "x"]
SPATIAL_IMAGE_DIMS = ["t", "c"] + SPATIAL_DIMS

DEFAULT_SPATIAL_CHUNKSIZES_3D = {"z": 256, "y": 256, "x": 256}
DEFAULT_SPATIAL_CHUNKSIZES_2D = {"y": 2048, "x": 2048}


def get_default_spatial_chunksizes(ndim: int):
    assert ndim in [2, 3]
    if ndim == 2:
        return DEFAULT_SPATIAL_CHUNKSIZES_2D
    elif ndim == 3:
        return DEFAULT_SPATIAL_CHUNKSIZES_3D


def get_sim_from_array(
    array: ArrayLike,
    dims: Optional[Union[list, tuple]] = None,
    scale: Optional[dict] = None,
    translation: Optional[dict] = None,
    affine: Optional[xr.DataArray] = None,
    transform_key: str = DEFAULT_TRANSFORM_KEY,
    c_coords: Optional[Union[list, tuple, ArrayLike]] = None,
    t_coords: Optional[Union[list, tuple, ArrayLike]] = None,
):
    """
    Get a spatial-image (multiview-stitcher flavor)
    from an array-like object.

    Parameters
    ----------
    array : ArrayLike
        Image data
    dims : Optional[Union[list, tuple]], optional
        Image dimension. Subset of ('t', 'c', 'z', 'y', 'x')
    scale : Optional[dict], optional
        Pixel spacing, e.g. {'z': 1.0, 'y': 0.3, 'x': 0.3}
    translation : Optional[dict], optional
        Image offset {'z': 50.0, 'y': 50. 'x': 50.}
    affine : Optional[xr.DataArray], optional
        Affine transform, e.g. xr.DataArray(np.eye(4), dims=["x_in", "x_out"])
    transform_key : str, optional
        By default DEFAULT_TRANSFORM_KEY
    c_coords : Optional[Union[list, tuple, ArrayLike]], optional
        Channel coordinates, e.g. ['DAPI', 'GFP', 'RFP']
    t_coords : Optional[Union[list, tuple, ArrayLike]], optional
        Time coordinates, e.g. [0.0, 0.2, 0.4, 0.6, 0.8]

    Returns
    -------
    spatial_image.SpatialImage
        spatial-image with multiview-stitcher flavor
        (SpatialImage + affine transform attributes)
    """

    if dims is None:
        dims = ["t", "c", "z", "y", "x"][-array.ndim :]
    else:
        assert len(dims) == array.ndim

    xim = xr.DataArray(
        array,
        dims=dims,
    )

    nsdims = ["c", "t"]
    for nsdim in nsdims:
        if nsdim not in xim.dims:
            xim = xim.expand_dims([nsdim])

    # transpose to dim order supported by spatial-image
    new_dims = [dim for dim in SPATIAL_IMAGE_DIMS if dim in xim.dims]
    if new_dims != xim.dims:
        xim = xim.transpose(*new_dims)

    spatial_dims = [dim for dim in xim.dims if dim in SPATIAL_DIMS]
    ndim = len(spatial_dims)

    if not isinstance(array, da.Array):
        xim = xim.chunk(
            {dim: 1 for dim in nsdims} | get_default_spatial_chunksizes(ndim)
        )

    if scale is None:
        scale = {dim: 1 for dim in spatial_dims}

    if translation is None:
        translation = {dim: 0 for dim in spatial_dims}

    sim = si.to_spatial_image(
        xim.data,
        dims=xim.dims,
        scale=scale,
        translation=translation,
        c_coords=c_coords,
        t_coords=t_coords,
    )

    if affine is None:
        affine_xr = param_utils.identity_transform(ndim, t_coords=None)
    else:
        affine_xr = param_utils.affine_to_xaffine(affine)

    set_sim_affine(
        sim,
        affine_xr,
        transform_key=transform_key,
    )

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


def extend_stack_props(stack_props, extend_by):
    """
    Extend stack properties by a given extent along each spatial dimension.

    Parameters
    ----------
    stack_props
    extend_by : dict or float
    """
    sdims = [
        sdim
        for sdim in SPATIAL_DIMS
        if sdim in list(stack_props["spacing"].keys())
    ]
    if not isinstance(extend_by, dict):
        extend_by = {dim: extend_by for dim in sdims}

    for dim, val in extend_by.items():
        stack_props["shape"][dim] += int(
            np.ceil(2 * val / stack_props["spacing"][dim])
        )
        stack_props["origin"][dim] -= val

    return stack_props


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

    spacing = get_spacing_from_sim(sim, asarray=False)
    origin = get_origin_from_sim(sim, asarray=False)
    shape = get_shape_from_sim(sim, asarray=False)

    center = np.array(
        [
            origin[dim] + spacing[dim] * (shape[dim] - 1) / 2
            for dim in get_spatial_dims_from_sim(sim)
        ]
    )

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


def combine_attrs_func(x, context=None):
    """
    Function to be passed to xarray methods
    to manage combining multiview-stitcher flavoured
    SpatialImage attrributes.

    Example:
    ```
    xr.concat(..., combine_attrs=combine_attrs_func))
    ```
    """
    return {
        k: {
            transform_key: xr.concat(
                [v["transforms"][transform_key] for v in x], dim="t"
            )
            for transform_key in x[0]["transforms"]
        }
        for k, _ in x[0].items()
        if k == "transforms"
    }


def concat(sims, dim, **kwargs):
    """
    Concatenate multiview-stitcher flavoured SpatialImages.

    Same as xr.concat but with handling of
    transform_keys in attributes.

    Parameters
    ----------
    sims : spatial_image.SpatialImage
        multiview-stitcher flavor spatial images
    dim : str
        dim to concatenate over
    """

    return xr.concat(sims, dim=dim, combine_attrs=combine_attrs_func, **kwargs)


def combine_by_coords(sims, **kwargs):
    """
    Combine multiview-stitcher flavoured SpatialImages
    by coordinates.

    Same as xr.combine_by_coords but with handling of
    transform_keys in attributes.

    Parameters
    ----------
    sims : spatial_image.SpatialImage
        multiview-stitcher flavor spatial images
    """

    return xr.combine_by_coords(
        sims, combine_attrs=combine_attrs_func, **kwargs
    )


def max_project_sim(sim, dim="z"):
    """
    Max project a multiview-stitcher flavoured SpatialImage.

    Parameters
    ----------
    sim : spatial_image.SpatialImage
        multiview-stitcher flavor spatial image
    dim : str, optional
        dimension to project over, by default "z"

    Returns
    -------
    spatial_image.SpatialImage
        maximum projected spatial image
    """

    sim = sim.max(dim=dim, keep_attrs=True).copy(deep=True)

    # project transforms
    for transform_key in sim.attrs["transforms"]:
        affine = get_affine_from_sim(sim, transform_key)
        affine = affine.sel(
            {
                pdim: [
                    sdim for sdim in affine.coords[pdim].values if sdim != dim
                ]
                for pdim in ["x_in", "x_out"]
            }
        )
        set_sim_affine(sim, affine, transform_key)

    return sim
