import copy
import threading
from typing import Optional, Union

import dask.array as da
import numpy as np
import xarray as xr
import zarr
from numpy._typing import ArrayLike
from xarray.backends import BackendArray
from xarray.core import indexing

from multiview_stitcher import param_utils

DEFAULT_TRANSFORM_KEY = "affine_metadata"

SPATIAL_DIMS = ["z", "y", "x"]
SPATIAL_IMAGE_DIMS = ["t", "c"] + SPATIAL_DIMS

DEFAULT_SPATIAL_CHUNKSIZES_3D = {"z": 256, "y": 256, "x": 256}
DEFAULT_SPATIAL_CHUNKSIZES_2D = {"y": 2048, "x": 2048}
HTTP_ZARR_ASYNC_CONCURRENCY = 4
HTTP_ZARR_MAX_SIMULTANEOUS_MATERIALIZATIONS = 1
_HTTP_ZARR_MATERIALIZATION_SEMAPHORE = threading.BoundedSemaphore(
    HTTP_ZARR_MAX_SIMULTANEOUS_MATERIALIZATIONS
)


class ZarrLazyBackendArray(BackendArray):
    # Wrap a raw zarr array in xarray's BackendArray protocol so xarray can
    # keep reads lazy until we explicitly chunk the selected region with dask.
    def __init__(self, zarray):
        self.zarray = zarray
        self.shape = zarray.shape
        self.dtype = np.dtype(zarray.dtype)

    def _raw_indexing_method(self, key):
        return self.zarray[key]

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )


class SingletonExpandedBackendArray(BackendArray):
    # xarray.expand_dims materializes some backend arrays. This wrapper presents
    # leading singleton axes virtually, so we can add missing t/c dims without
    # touching the underlying zarr-backed payload.
    def __init__(self, array, singleton_axes):
        self.array = array
        self.singleton_axes = tuple(singleton_axes)
        self.dtype = np.dtype(array.dtype)

        inner_shape = iter(array.shape)
        self.shape = tuple(
            1 if axis in self.singleton_axes else next(inner_shape)
            for axis in range(len(array.shape) + len(self.singleton_axes))
        )

    def _raw_indexing_method(self, key):
        if not isinstance(key, tuple):
            key = (key,)

        key = key + (slice(None),) * (len(self.shape) - len(key))

        inner_key = []
        expand_axes = []
        output_axis = 0

        for axis, item in enumerate(key):
            if axis in self.singleton_axes:
                if not isinstance(item, (int, np.integer)):
                    expand_axes.append(output_axis)
                    output_axis += 1
                continue

            inner_key.append(item)
            if not isinstance(item, (int, np.integer)):
                output_axis += 1

        inner_key = tuple(inner_key)
        if hasattr(self.array, "_updated_key"):
            result = np.asarray(self.array[indexing.BasicIndexer(inner_key)])
        else:
            result = np.asarray(self.array[inner_key])

        for axis in expand_axes:
            result = np.expand_dims(result, axis=axis)

        return result

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )


def _zarr_array_to_dataarray(
    zarray,
    dims,
    coords=None,
    name=None,
    attrs=None,
):
    # Preserve zarr-backed laziness when constructing an xarray.DataArray from a
    # plain zarr.Array, and propagate chunk hints for the later dask conversion.
    backend = ZarrLazyBackendArray(zarray)
    lazy_data = indexing.LazilyIndexedArray(backend)
    var = xr.Variable(
        dims,
        lazy_data,
        attrs={} if attrs is None else dict(attrs),
    )
    xim = xr.DataArray(var, coords=coords, name=name)

    zchunks = getattr(zarray, "chunks", None)
    if zchunks is not None:
        xim.attrs["_zarr_chunks"] = tuple(zchunks)
        xim.encoding["preferred_chunks"] = {
            dim: chunk for dim, chunk in zip(dims, zchunks)
        }

    return xim


def _iter_backend_arrays(array):
    current = array
    seen = set()

    while current is not None and id(current) not in seen:
        yield current
        seen.add(id(current))

        next_current = None
        for attr in ("array", "_array", "zarray"):
            candidate = getattr(current, attr, None)
            if candidate is not None and candidate is not current:
                next_current = candidate
                break

        current = next_current


def is_xarray_zarr_backed(xim):
    if not isinstance(xim, xr.DataArray):
        return False

    internal = getattr(xim.variable, "_data", None)
    if internal is None:
        return False

    for candidate in _iter_backend_arrays(internal):
        if isinstance(candidate, zarr.Array):
            return True

        candidate_type = type(candidate)
        if candidate_type.__module__.startswith("xarray.backends.zarr"):
            return True

        if hasattr(candidate, "zarray") and isinstance(candidate.zarray, zarr.Array):
            return True

    return False


def _get_xarray_zarr_array(xim):
    if not isinstance(xim, xr.DataArray):
        return None

    internal = getattr(xim.variable, "_data", None)
    if internal is None:
        return None

    for candidate in _iter_backend_arrays(internal):
        if isinstance(candidate, zarr.Array):
            return candidate

        if hasattr(candidate, "zarray") and isinstance(candidate.zarray, zarr.Array):
            return candidate.zarray

    return None


def _zarr_array_uses_http_store(zarray):
    if zarray is None:
        return False

    store = getattr(zarray, "store", None)
    fs = getattr(store, "fs", None)
    protocol = getattr(fs, "protocol", ())

    if isinstance(protocol, str):
        protocol = (protocol,)

    return any(proto in {"http", "https"} for proto in protocol)


def _materialize_xarray_zarr_backend(xim, max_retries=3):
    # Public HTTP-backed stores have been prone to disconnects when one slice
    # fans out into several concurrent chunk fetches inside zarr.
    from aiohttp.client_exceptions import ServerDisconnectedError

    zarray = _get_xarray_zarr_array(xim)
    backend_data = _get_backend_data(xim)

    if _zarr_array_uses_http_store(zarray):
        last_error = None
        for _ in range(max_retries):
            try:
                with _HTTP_ZARR_MATERIALIZATION_SEMAPHORE:
                    with zarr.config.set(
                        {"async.concurrency": HTTP_ZARR_ASYNC_CONCURRENCY}
                    ):
                        return np.asarray(backend_data)
            except ServerDisconnectedError as exc:
                last_error = exc

        raise last_error

    return np.asarray(backend_data)


def _get_backend_data(xim):
    internal = getattr(xim.variable, "_data", None)
    if internal is not None:
        return internal

    return xim.data


def is_dask_backed_dataarray(xim):
    return isinstance(_get_backend_data(xim), da.Array)


def _get_preferred_chunks(xim):
    preferred_chunks = xim.encoding.get("preferred_chunks")

    if preferred_chunks is None:
        zarr_chunks = xim.attrs.get("_zarr_chunks")
        if zarr_chunks is not None:
            preferred_chunks = {
                dim: chunk for dim, chunk in zip(xim.dims, zarr_chunks)
            }

    if preferred_chunks is None:
        spatial_dims = [dim for dim in xim.dims if dim in SPATIAL_DIMS]
        spatial_chunks = get_default_spatial_chunksizes(len(spatial_dims))
        preferred_chunks = {
            dim: (1 if dim in ["c", "t"] else spatial_chunks[dim])
            for dim in xim.dims
        }

    return {
        dim: min(int(preferred_chunks[dim]), xim.sizes[dim])
        for dim in xim.dims
        if dim in preferred_chunks
    }


def _expand_with_singleton_dims_lazily(xim, missing_dims):
    # Add missing non-spatial dims without forcing xarray to realize the zarr
    # backend; registration/fusion convert to dask only after slicing.
    backend = SingletonExpandedBackendArray(
        _get_backend_data(xim),
        singleton_axes=tuple(range(len(missing_dims))),
    )
    lazy_data = indexing.LazilyIndexedArray(backend)
    dims = tuple(missing_dims) + tuple(xim.dims)
    expanded = xr.DataArray(
        xr.Variable(dims, lazy_data),
        coords={dim: xim.coords[dim] for dim in xim.coords if dim in xim.dims},
        name=xim.name,
    )

    preferred_chunks = dict(xim.encoding.get("preferred_chunks", {}))
    preferred_chunks.update({dim: 1 for dim in missing_dims})
    if preferred_chunks:
        expanded.encoding["preferred_chunks"] = {
            dim: preferred_chunks[dim]
            for dim in expanded.dims
            if dim in preferred_chunks
        }

    if "_zarr_chunks" in xim.attrs:
        expanded.attrs["_zarr_chunks"] = xim.attrs["_zarr_chunks"]

    return expanded


def ensure_dask_backed_dataarray(xim):
    if is_xarray_zarr_backed(xim):
        # Convert only the already-selected region into dask to avoid building a
        # large graph from slicing the full input array first.
        return xim.chunk(_get_preferred_chunks(xim))

    if is_dask_backed_dataarray(xim):
        return xim

    return xim


def _copy_chunk_hints(source, target):
    preferred_chunks = source.encoding.get("preferred_chunks")
    if preferred_chunks is not None:
        target.encoding["preferred_chunks"] = dict(preferred_chunks)

    if "_zarr_chunks" in source.attrs:
        target.attrs["_zarr_chunks"] = source.attrs["_zarr_chunks"]


def _get_axis_coords(dim, size, scale, translation):
    return translation + scale * np.arange(size, dtype=float)


def to_spatial_image(
    data,
    dims=None,
    scale=None,
    translation=None,
    c_coords=None,
    t_coords=None,
):
    if scale is None or translation is None:
        raise ValueError("scale and translation must be provided")

    if isinstance(data, xr.DataArray):
        name = data.name
        data = _get_backend_data(data)
    else:
        name = None

    if dims is None:
        dims = SPATIAL_DIMS[-data.ndim :]

    dims = tuple(dims)
    shape = data.shape
    coords = {}

    for axis, dim in enumerate(dims):
        size = shape[axis]
        if dim in SPATIAL_DIMS:
            coords[dim] = _get_axis_coords(
                dim,
                size,
                scale=scale[dim],
                translation=translation[dim],
            )
        elif dim == "c":
            coords[dim] = (
                np.asarray(c_coords)
                if c_coords is not None
                else np.arange(size)
            )
        elif dim == "t":
            coords[dim] = (
                np.asarray(t_coords)
                if t_coords is not None
                else np.arange(size)
            )

    return xr.DataArray(
        xr.Variable(dims, data),
        coords=coords,
        name=name,
    )


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

    if isinstance(array, xr.DataArray):
        xim = array.copy(deep=False)
        if dims is None:
            dims = list(xim.dims)
        else:
            assert len(dims) == xim.ndim
            if tuple(dims) != tuple(xim.dims):
                xim = xim.transpose(*dims)
    else:
        if dims is None:
            dims = ["t", "c", "z", "y", "x"][-array.ndim :]
        else:
            assert len(dims) == array.ndim

        if isinstance(array, zarr.Array):
            xim = _zarr_array_to_dataarray(array, dims=dims)
        else:
            xim = xr.DataArray(
                array,
                dims=dims,
            )

    if c_coords is None and "c" in xim.coords:
        c_coords = xim.coords["c"].values

    if t_coords is None and "t" in xim.coords:
        t_coords = xim.coords["t"].values

    nsdims = ["c", "t"]
    missing_nsdims = [nsdim for nsdim in reversed(nsdims) if nsdim not in xim.dims]
    if missing_nsdims:
        if is_xarray_zarr_backed(xim):
            # Keep missing singleton axes virtual for zarr-backed inputs.
            xim = _expand_with_singleton_dims_lazily(xim, missing_nsdims)
        else:
            for nsdim in nsdims:
                if nsdim not in xim.dims:
                    xim = xim.expand_dims([nsdim])

    # transpose to dim order supported by spatial-image
    new_dims = [dim for dim in SPATIAL_IMAGE_DIMS if dim in xim.dims]
    if new_dims != xim.dims:
        xim = xim.transpose(*new_dims)

    spatial_dims = [dim for dim in xim.dims if dim in SPATIAL_DIMS]
    ndim = len(spatial_dims)

    if (
        not isinstance(getattr(xim.variable, "_data", None), da.Array)
        and not is_xarray_zarr_backed(xim)
    ):
        xim = xim.chunk(
            {dim: 1 for dim in nsdims} | get_default_spatial_chunksizes(ndim)
        )

    if scale is None:
        scale = {dim: 1 for dim in spatial_dims}

    if translation is None:
        translation = {dim: 0 for dim in spatial_dims}

    sim = to_spatial_image(
        xim,
        dims=xim.dims,
        scale=scale,
        translation=translation,
        c_coords=c_coords,
        t_coords=t_coords,
    )

    _copy_chunk_hints(xim, sim)

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


def get_extent_from_sim(sim):
    """
    Get extent from sim, calculated as the span between the first and last
    coordinate for each spatial dimension.
    """
    sp = get_stack_properties_from_sim(sim)
    extent = {
        dim: (sp["shape"][dim] - 1) * sp["spacing"][dim]
        for dim in sp["shape"]
    }
    return extent


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

    sim = to_spatial_image(
        xim,
        dims=xim.dims,
        scale=spacing,
        translation=origin,
        t_coords=xim.coords["t"] if "t" in xim.coords else None,
        c_coords=xim.coords["c"] if "c" in xim.coords else None,
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
