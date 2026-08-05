"""
Dependency-free reading and writing of OME-Zarr (NGFF) multiscales metadata.

``ngff-zarr`` and ``ome-zarr-py`` are the reference implementations used by
multiview-stitcher on CPython. Neither is installable in Pyodide without
pulling in wheels that have no WebAssembly build, so this module provides a
small, self-contained equivalent for the subset of NGFF v0.4 / v0.5 that
multiview-stitcher reads and writes:

* parsing ``multiscales`` metadata into the duck-typed shape that
  :func:`multiview_stitcher.ngff_utils.ngff_multiscales_to_msim` consumes, and
* writing ``multiscales`` metadata in the same layout that
  ``ome_zarr.writer.write_multiscales_metadata`` produces.

``ngff_utils`` prefers the reference implementations whenever they import, and
falls back to this module otherwise, so both paths stay exercised.

Sources may be given either as a path/URL string or as an already-constructed
zarr store (or plain mapping). The latter is what the browser runtime uses to
route reads through a service worker.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import zarr

# Imported for its side effect as much as anything: reading an OME-Zarr
# written by another tool can need codec metadata this numcodecs would
# otherwise refuse. See `_zarr_compat.register_compatible_codecs`.
from multiview_stitcher import _zarr_compat  # noqa: F401

# NGFF axis "type" values that are not spatial.
_NON_SPATIAL_AXIS_TYPES = {"time", "channel"}


def _is_pathlike(source):
    return isinstance(source, (str, os.PathLike))


def open_zarr_group(source, mode="r", **kwargs):
    """Open a zarr group from a path/URL or from a store-like object."""
    if _is_pathlike(source):
        return zarr.open_group(str(source), mode=mode, **kwargs)
    return zarr.open_group(store=source, mode=mode, **kwargs)


def open_zarr_array(source, path, mode="r", **kwargs):
    """Open a zarr array below ``source`` at the relative ``path``."""
    if _is_pathlike(source):
        return zarr.open_array(
            os.path.join(str(source), str(path)), mode=mode, **kwargs
        )
    return zarr.open_array(
        store=source, path=str(path), mode=mode, **kwargs
    )


@dataclass
class NgffImage:
    """Minimal stand-in for ``ngff_zarr.NgffImage``."""

    data: Any
    dims: tuple
    scale: dict
    translation: dict
    name: str = "image"
    axes_units: dict = field(default_factory=dict)


@dataclass
class NgffDataset:
    """Minimal stand-in for ``ngff_zarr.Dataset``."""

    path: str
    coordinateTransformations: list  # noqa: N815 - NGFF spelling


@dataclass
class NgffMetadata:
    """Minimal stand-in for ``ngff_zarr.Metadata``."""

    axes: list
    datasets: list
    name: str = "image"
    version: str = "0.4"
    coordinateTransformations: Optional[list] = None  # noqa: N815
    omero: Any = None


@dataclass
class NgffMultiscales:
    """Minimal stand-in for ``ngff_zarr.Multiscales``."""

    images: list
    metadata: NgffMetadata
    scale_factors: list = field(default_factory=list)


def _normalize_axes(axes, ndim):
    """Return NGFF axis dicts for any supported ``axes`` spelling.

    NGFF v0.1/v0.2 have no axes at all, v0.3 uses a list of names, and
    v0.4/v0.5 use a list of dicts.
    """
    if not axes:
        # Pre-0.3 files: fall back to the canonical axis order.
        default = ["t", "c", "z", "y", "x"][-ndim:]
        axes = list(default)

    normalized = []
    for axis in axes:
        if isinstance(axis, str):
            axis = {"name": axis}
        name = axis["name"]
        axis_type = axis.get(
            "type",
            "time"
            if name == "t"
            else ("channel" if name == "c" else "space"),
        )
        normalized.append(
            {
                "name": name,
                "type": axis_type,
                **({"unit": axis["unit"]} if axis.get("unit") else {}),
            }
        )
    return normalized


def _transform_vectors(coordinate_transformations, ndim):
    """Reduce a coordinateTransformations list to (scale, translation) vectors."""
    scale = np.ones(ndim, dtype=float)
    translation = np.zeros(ndim, dtype=float)

    for transform in coordinate_transformations or []:
        transform_type = transform.get("type")
        if transform_type == "scale":
            values = np.asarray(transform["scale"], dtype=float)
            scale = scale * values
            translation = translation * values
        elif transform_type == "translation":
            translation = translation + np.asarray(
                transform["translation"], dtype=float
            )
        elif transform_type == "identity":
            continue
        else:
            raise ValueError(
                f"Unsupported NGFF coordinateTransformation type "
                f"'{transform_type}'."
            )

    return scale, translation


def read_multiscales_attrs(source):
    """Return ``(multiscales_dict, ngff_version, group_attrs)`` for a source."""
    root = open_zarr_group(source, mode="r")
    attrs = dict(root.attrs)

    if "ome" in attrs:
        ome = attrs["ome"]
        version = str(ome.get("version", "0.5"))
        multiscales = ome["multiscales"]
    elif "multiscales" in attrs:
        multiscales = attrs["multiscales"]
        version = str(multiscales[0].get("version", "0.4"))
    else:
        raise ValueError(f"No OME-Zarr multiscales metadata found in {source}.")

    return multiscales[0], version, attrs


def read_ngff_multiscales(source, array_opener=None):
    """Parse NGFF multiscales metadata into duck-typed multiscales objects.

    Parameters
    ----------
    source : str, os.PathLike or store-like
        OME-Zarr root.
    array_opener : callable, optional
        ``(source, dataset_path) -> array``. Defaults to opening the on-disk
        zarr arrays read-only, which is what the zarr-backed read path wants.
    """
    if array_opener is None:
        array_opener = open_zarr_array

    multiscales, version, _ = read_multiscales_attrs(source)

    datasets = multiscales["datasets"]
    arrays = [array_opener(source, dataset["path"]) for dataset in datasets]
    ndim = arrays[0].ndim

    axes = _normalize_axes(multiscales.get("axes"), ndim)
    if len(axes) != ndim:
        raise ValueError(
            f"NGFF axes ({len(axes)}) do not match array rank ({ndim})."
        )
    dims = tuple(axis["name"] for axis in axes)
    axes_units = {
        axis["name"]: axis["unit"] for axis in axes if axis.get("unit")
    }

    # Multiscales-level transforms apply after the per-dataset ones.
    outer_scale, outer_translation = _transform_vectors(
        multiscales.get("coordinateTransformations"), ndim
    )

    images = []
    for dataset, array in zip(datasets, arrays):
        scale, translation = _transform_vectors(
            dataset.get("coordinateTransformations"), ndim
        )
        scale = scale * outer_scale
        translation = translation * outer_scale + outer_translation
        images.append(
            NgffImage(
                data=array,
                dims=dims,
                scale={
                    dim: float(value) for dim, value in zip(dims, scale)
                },
                translation={
                    dim: float(value) for dim, value in zip(dims, translation)
                },
                name=str(multiscales.get("name", "image")),
                axes_units=axes_units,
            )
        )

    sdims = [
        axis["name"]
        for axis in axes
        if axis["type"] not in _NON_SPATIAL_AXIS_TYPES
    ]
    scale_factors = [
        {
            sdim: int(
                round(
                    images[0].data.shape[dims.index(sdim)]
                    / images[iscale].data.shape[dims.index(sdim)]
                )
            )
            for sdim in sdims
        }
        for iscale in range(1, len(images))
    ]

    metadata = NgffMetadata(
        axes=axes,
        datasets=[
            NgffDataset(
                path=str(dataset["path"]),
                coordinateTransformations=dataset.get(
                    "coordinateTransformations", []
                ),
            )
            for dataset in datasets
        ],
        name=str(multiscales.get("name", "image")),
        version=version,
        coordinateTransformations=multiscales.get(
            "coordinateTransformations"
        ),
    )

    return NgffMultiscales(
        images=images, metadata=metadata, scale_factors=scale_factors
    )


def write_multiscales_metadata(group, axes, datasets, ngff_version="0.4"):
    """Write NGFF ``multiscales`` metadata into an open zarr ``group``.

    Mirrors ``ome_zarr.writer.write_multiscales_metadata`` for the versions
    multiview-stitcher writes: v0.4 keeps ``multiscales`` (including its
    ``version`` key) at the top level of the group attributes, while v0.5
    nests ``multiscales`` and ``version`` inside an ``ome`` attribute.
    """
    multiscale = {
        "datasets": [dict(dataset) for dataset in datasets],
        "name": group.name,
        "axes": [dict(axis) for axis in axes],
    }

    if str(ngff_version).startswith("0.4"):
        multiscale["version"] = str(ngff_version)
        group.attrs["multiscales"] = [multiscale]
        return

    ome = dict(group.attrs.get("ome", {}))
    ome["version"] = str(ngff_version)
    ome["multiscales"] = [multiscale]
    group.attrs["ome"] = ome
