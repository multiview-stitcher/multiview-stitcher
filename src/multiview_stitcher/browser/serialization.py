"""
JSON encoding of the small objects exchanged between Python and JavaScript.

Only *metadata* crosses this boundary: dataset descriptions, user options,
registration results and stack properties. Image data never does - it stays
inside the Python heap of the worker that owns it and leaves only as encoded
zarr chunk bytes (see :mod:`multiview_stitcher.browser.session`).
"""

import numpy as np
import xarray as xr

from multiview_stitcher import msi_utils, param_utils
from multiview_stitcher import spatial_image_utils as si_utils


def to_jsonable(obj):
    """Recursively convert numpy / xarray scalars and containers to JSON types."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, xr.DataArray):
        return to_jsonable(obj.values)
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(value) for value in obj]
    return str(obj)


# ---------------------------------------------------------------------------
# Affine transform parameters
# ---------------------------------------------------------------------------


def dataarray_to_json(array):
    """Serialise a numeric ``xr.DataArray`` with its dims and coordinates.

    Used for every labelled array that crosses the worker boundary: affine
    transforms, registration qualities and overlap bounding boxes.
    """
    array = xr.DataArray(array)
    return {
        "dims": [str(dim) for dim in array.dims],
        "coords": {
            str(dim): to_jsonable(array.coords[dim].values)
            for dim in array.dims
            if dim in array.coords
        },
        "data": to_jsonable(array.values),
    }


def dataarray_from_json(payload):
    """Inverse of :func:`dataarray_to_json`."""
    if payload is None:
        return None

    dims = [str(dim) for dim in payload["dims"]]
    coords = {
        str(dim): list(values)
        for dim, values in (payload.get("coords") or {}).items()
        if dim in dims
    }
    return xr.DataArray(
        np.asarray(payload["data"], dtype=float), dims=dims, coords=coords
    )


#: Affine transforms are plain labelled arrays; keep the descriptive names.
affine_to_json = dataarray_to_json
affine_from_json = dataarray_from_json


def pairwise_result_to_json(param_ds):
    """Serialise one pairwise registration result (an ``xr.Dataset``)."""
    return {
        key: dataarray_to_json(param_ds[key])
        for key in ("transform", "quality", "bbox")
    }


def pairwise_result_from_json(payload):
    """Inverse of :func:`pairwise_result_to_json`.

    Returns a plain dict, which is all
    ``registration._assign_pairwise_registrations`` needs.
    """
    return {
        key: dataarray_from_json(payload[key])
        for key in ("transform", "quality", "bbox")
    }


def params_to_json(params):
    """Serialise a list of per-view affine transforms."""
    return [affine_to_json(param) for param in params]


def params_from_json(payload):
    """Inverse of :func:`params_to_json`."""
    return [affine_from_json(param) for param in payload]


# ---------------------------------------------------------------------------
# Stack properties
# ---------------------------------------------------------------------------


def stack_properties_to_json(stack_properties):
    return {
        key: {
            str(dim): (
                int(value) if key == "shape" else float(value)
            )
            for dim, value in stack_properties[key].items()
        }
        for key in ("origin", "spacing", "shape")
        if key in stack_properties
    }


def stack_properties_from_json(payload):
    if payload is None:
        return None
    return {
        "origin": {
            str(dim): float(value)
            for dim, value in payload["origin"].items()
        },
        "spacing": {
            str(dim): float(value)
            for dim, value in payload["spacing"].items()
        },
        "shape": {
            str(dim): int(value) for dim, value in payload["shape"].items()
        },
    }


# ---------------------------------------------------------------------------
# Image metadata
# ---------------------------------------------------------------------------


def _transform_keys(msim):
    """Names of the extrinsic coordinate systems attached to an msim."""
    return sorted(
        str(name)
        for name in msim["scale0"].data_vars
        if str(name) != "image"
    )


def msim_metadata(msim, name=None):
    """Describe an msim for the UI: geometry, channels and transform keys.

    Deliberately small and lazy: nothing here touches image data.
    """
    scale_keys = msi_utils.get_sorted_scale_keys(msim)
    sim0 = msi_utils.get_sim_from_msim(msim, scale=scale_keys[0])
    sdims = si_utils.get_spatial_dims_from_sim(sim0)

    levels = []
    for scale_key in scale_keys:
        sim = msi_utils.get_sim_from_msim(msim, scale=scale_key)
        levels.append(
            {
                "key": str(scale_key),
                "shape": {str(dim): int(sim.sizes[dim]) for dim in sim.dims},
                "spacing": to_jsonable(si_utils.get_spacing_from_sim(sim)),
                "origin": to_jsonable(si_utils.get_origin_from_sim(sim)),
            }
        )

    metadata = {
        "name": name,
        "dims": [str(dim) for dim in sim0.dims],
        "spatial_dims": [str(dim) for dim in sdims],
        "ndim": len(sdims),
        "dtype": str(sim0.dtype),
        "levels": levels,
        "transform_keys": _transform_keys(msim),
    }

    for dim in ("t", "c"):
        if dim in sim0.dims:
            metadata[f"{dim}_coords"] = [
                str(value) for value in sim0.coords[dim].values
            ]

    return metadata


def transform_from_msim_json(msim, transform_key):
    """Serialise the transform attached to ``transform_key`` of an msim."""
    return affine_to_json(
        msi_utils.get_transform_from_msim(msim, transform_key=transform_key)
    )


def apply_transforms(msim, transforms, base_transform_key=None):
    """Attach serialised transforms to an msim.

    ``transforms`` maps a transform key to its JSON affine. This is how a
    compute worker reproduces the state of the session worker without ever
    receiving image data.
    """
    for transform_key, payload in (transforms or {}).items():
        xaffine = affine_from_json(payload)
        if xaffine is None:
            ndim = msi_utils.get_ndim(msim)
            xaffine = param_utils.identity_transform(ndim)
        msi_utils.set_affine_transform(
            msim,
            xaffine,
            transform_key=transform_key,
            base_transform_key=base_transform_key,
        )
    return msim
