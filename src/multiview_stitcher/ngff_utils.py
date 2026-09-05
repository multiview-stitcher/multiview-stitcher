import atexit
import asyncio
from dataclasses import asdict
from functools import partial
import json
import math
import os, shutil
import signal
import threading
import uuid
from copy import deepcopy

import dask
import numpy as np
import zarr
from tqdm import tqdm
from dask import array as da
import dask.diagnostics
from xarray import DataTree

import ngff_zarr

from multiview_stitcher import (
    _ngff_v06,
    _zarr_compat,
    msi_utils,
    param_utils,
    misc_utils,
)
from multiview_stitcher import spatial_image_utils as si_utils

# Original axes of an image read from NGFF.  Spatial images deliberately add
# missing singleton ``t``/``c`` dimensions, but a viewer transform attached to
# the original OME-Zarr must still have the rank of the array on disk.
NGFF_AXES_UNITS_ATTR = "_multiview_stitcher_ngff_axes_units"

NGFF_SOURCE_DIMS_ATTR = "_multiview_stitcher_ngff_source_dims"

# Calibration of the NGFF time axis.  Spatial images carry their spatial
# calibration in the coordinates themselves, but ``t`` coordinates are frame
# indices, so a non-unity NGFF time scale has nowhere else to live.  Losing it
# is not merely cosmetic: a viewer reading the original store sees the stored
# time scale, so an image served or written as if the scale were 1 no longer
# lines up with it along ``t``.
NGFF_TIME_TRANSFORM_ATTR = "_multiview_stitcher_ngff_time_transform"

DEFAULT_NGFF_TIME_TRANSFORM = {
    "scale": 1.0,
    "translation": 0.0,
    "unit": None,
}


def _ngff_time_transform_of(ngff_im):
    """The time calibration an ``ngff_zarr`` image declares for its ``t`` axis."""
    return {
        "scale": float((ngff_im.scale or {}).get("t", 1.0)),
        "translation": float((ngff_im.translation or {}).get("t", 0.0)),
        "unit": (ngff_im.axes_units or {}).get("t"),
    }


def _time_transform_holders(image):
    """The images an NGFF time calibration is stored on.

    A multiscale image keeps one copy per resolution level, mirroring how the
    NGFF source dims are stored: ``msi_utils`` strips image attrs when it
    assembles a DataTree, so the attribute has to be reattached per scale.
    """
    if msi_utils.is_msim(image):
        return [
            image[f"{scale_key}/image"]
            for scale_key in msi_utils.get_sorted_scale_keys(image)
        ]
    return [image]


def get_ngff_time_transform(image):
    """The NGFF time scale, translation and unit carried by ``image``.

    Returns the identity calibration for images that carry none, so callers
    can use the result unconditionally.
    """
    holders = _time_transform_holders(image)
    stored = holders[0].attrs.get(NGFF_TIME_TRANSFORM_ATTR) if holders else None
    return {**DEFAULT_NGFF_TIME_TRANSFORM, **(stored or {})}


def set_ngff_time_transform(image, time_transform):
    """Attach an NGFF time calibration to a spatial or multiscale image.

    An identity calibration is stored as the absence of the attribute, which
    keeps images that never had a time scale byte-identical to before.
    """
    time_transform = {
        **DEFAULT_NGFF_TIME_TRANSFORM,
        **(time_transform or {}),
    }
    for holder in _time_transform_holders(image):
        if time_transform == DEFAULT_NGFF_TIME_TRANSFORM:
            holder.attrs.pop(NGFF_TIME_TRANSFORM_ATTR, None)
        else:
            holder.attrs[NGFF_TIME_TRANSFORM_ATTR] = dict(time_transform)
    return image


def copy_ngff_time_transform(source, target):
    """Give ``target`` the time calibration of ``source``.

    A derived image - a fused stack, say - spans the same timepoints as the
    images it came from, but is built from a bare array, so the calibration
    has to be carried across rather than inherited.
    """
    return set_ngff_time_transform(target, get_ngff_time_transform(source))


def _drop_none_values(value):
    if isinstance(value, dict):
        return {
            key: _drop_none_values(val)
            for key, val in value.items()
            if val is not None
        }
    if isinstance(value, list):
        return [_drop_none_values(val) for val in value]
    return value


def _zarr_dtype(dtype):
    dtype = np.dtype(dtype)
    if dtype.byteorder == "=":
        if dtype.itemsize == 1:
            dtype = dtype.newbyteorder("|")
        else:
            dtype = dtype.newbyteorder("<" if np.little_endian else ">")
    return dtype.str


def _fill_value_for_dtype(dtype):
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.floating):
        return 0.0
    if np.issubdtype(dtype, np.integer):
        return 0
    if np.issubdtype(dtype, np.bool_):
        return False
    return 0


def _regular_chunks_from_dask_chunks(chunks, shape):
    chunk_shape = []
    for axis, (dim_chunks, dim_size) in enumerate(zip(chunks, shape)):
        if not dim_chunks:
            raise ValueError(f"Dimension {axis} has no chunk metadata.")

        regular_chunk = int(dim_chunks[0])
        if regular_chunk <= 0:
            raise ValueError(f"Invalid chunk size {regular_chunk}.")

        interior_chunks = dim_chunks[:-1]
        if any(int(chunk) != regular_chunk for chunk in interior_chunks):
            raise ValueError(
                "Virtual OME-Zarr serving requires regular chunks except "
                f"for final edge chunks; dimension {axis} has chunks "
                f"{dim_chunks}."
            )

        if int(dim_chunks[-1]) > regular_chunk:
            raise ValueError(
                "Virtual OME-Zarr serving requires final edge chunks to be "
                f"no larger than the declared chunk; dimension {axis} has "
                f"chunks {dim_chunks}."
            )

        chunk_shape.append(min(regular_chunk, int(dim_size)))

    return tuple(chunk_shape)


def _chunk_shape_from_sim(sim):
    data = si_utils._get_backend_data(sim)

    if hasattr(data, "chunks"):
        return _regular_chunks_from_dask_chunks(data.chunks, sim.shape)

    preferred_chunks = sim.encoding.get("preferred_chunks")
    if preferred_chunks is not None:
        return tuple(
            min(int(preferred_chunks[dim]), int(sim.sizes[dim]))
            for dim in sim.dims
        )

    return tuple(int(size) for size in sim.shape)


def _json_response_dict(obj):
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


class VirtualOMEZarr:
    """
    Read-only virtual OME-Zarr 0.4 / Zarr v2 hierarchy backed by an msim.

    Chunk requests are materialized directly from the source image with
    ``np.asarray(sim.isel(...).data)`` and no temporary Zarr store is written.
    """

    def __init__(self, msim, name="image", compressor=None, omero=None):
        if not msi_utils.is_msim(msim):
            raise TypeError("VirtualOMEZarr expects a MultiscaleSpatialImage.")

        self.msim = msim
        self.name = name
        self.compressor = compressor
        self.omero = omero if omero is not None else msim.attrs.get("omero")
        self.scale_keys = msi_utils.get_sorted_scale_keys(msim)
        if not self.scale_keys:
            raise ValueError("msim must contain at least one scale.")

        self.paths = [str(index) for index in range(len(self.scale_keys))]
        self.sims = [
            msi_utils.get_sim_from_msim(msim, scale=scale_key)
            for scale_key in self.scale_keys
        ]
        self.chunk_shapes = {
            path: _chunk_shape_from_sim(sim)
            for path, sim in zip(self.paths, self.sims)
        }

        self._root_zattrs = self._build_root_zattrs()

    def _build_root_zattrs(self):
        # Build OME-Zarr 0.4 .zattrs metadata directly from the sim coordinates
        # without calling ngff_zarr.to_multiscales, which would rechunk the dask
        # arrays and potentially trigger serialisation of large images to disk.
        _DIM_TYPE = {"t": "time", "c": "channel"}
        sdims = si_utils.get_spatial_dims_from_sim(self.sims[0])
        dims = self.sims[0].dims

        # A virtual store stands in for the image it was built from, so it has
        # to declare the same time calibration: a viewer places a natively
        # served store and a virtual one in a single coordinate space, and a
        # ``t`` scale of 1 here against a scaled one there would put them at
        # different points of the time axis.
        time_transform = get_ngff_time_transform(self.msim)
        _NSDIM_UNIT = {"t": time_transform["unit"]}
        _NSDIM_SCALE = {"t": float(time_transform["scale"])}
        _NSDIM_TRANSLATION = {"t": float(time_transform["translation"])}

        axes = [
            _drop_none_values({
                "name": dim,
                "type": _DIM_TYPE.get(dim, "space"),
                "unit": (
                    _NSDIM_UNIT.get(dim)
                    if dim in _DIM_TYPE
                    else "micrometer"
                ),
            })
            for dim in dims
        ]

        datasets = []
        for path, sim in zip(self.paths, self.sims):
            spacing = si_utils.get_spacing_from_sim(sim)
            origin = si_utils.get_origin_from_sim(sim)
            scale_values = [
                float(spacing[dim])
                if dim in sdims
                else _NSDIM_SCALE.get(dim, 1.0)
                for dim in dims
            ]
            translation_values = [
                float(origin[dim])
                if dim in sdims
                else _NSDIM_TRANSLATION.get(dim, 0.0)
                for dim in dims
            ]
            datasets.append({
                "path": path,
                "coordinateTransformations": [
                    {"type": "scale", "scale": scale_values},
                    {"type": "translation", "translation": translation_values},
                ],
            })

        metadata = {
            "version": "0.4",
            "name": self.name,
            "axes": axes,
            "datasets": datasets,
        }

        zattrs = {"multiscales": [metadata]}
        if self.omero is not None:
            zattrs["omero"] = (
                asdict(self.omero)
                if hasattr(self.omero, "__dataclass_fields__")
                else self.omero
            )

        return _drop_none_values(zattrs)

    def root_zgroup(self):
        return {"zarr_format": 2}

    def root_zattrs(self):
        return self._root_zattrs

    def array_zattrs(self, path):
        self._get_sim(path)
        return {}

    def array_zarray(self, path):
        sim = self._get_sim(path)
        chunk_shape = self.chunk_shapes[path]
        compressor = (
            self.compressor.get_config()
            if self.compressor is not None
            else None
        )

        return {
            "zarr_format": 2,
            "shape": [int(size) for size in sim.shape],
            "chunks": [int(size) for size in chunk_shape],
            "dtype": _zarr_dtype(sim.dtype),
            "compressor": compressor,
            "fill_value": _fill_value_for_dtype(sim.dtype),
            "order": "C",
            "filters": None,
            "dimension_separator": "/",
        }

    def consolidated_metadata(self):
        metadata = {
            ".zgroup": self.root_zgroup(),
            ".zattrs": self.root_zattrs(),
        }
        for path in self.paths:
            metadata[f"{path}/.zarray"] = self.array_zarray(path)
            metadata[f"{path}/.zattrs"] = self.array_zattrs(path)

        return {
            "zarr_consolidated_format": 1,
            "metadata": metadata,
        }

    def get_json_key(self, key):
        key = key.strip("/")

        if key == ".zgroup":
            return self.root_zgroup()
        if key == ".zattrs":
            return self.root_zattrs()
        if key == ".zmetadata":
            return self.consolidated_metadata()
        if key.endswith("/.zarray"):
            return self.array_zarray(key[: -len("/.zarray")])
        if key.endswith("/.zattrs"):
            return self.array_zattrs(key[: -len("/.zattrs")])

        raise KeyError(key)

    def chunk_content_length(self, path, chunk_key):
        """Return the byte length of a serialised chunk without materialising
        the underlying dask array.

        Returns ``None`` when a compressor is configured because the compressed
        size cannot be known without actually compressing the data.
        """
        if self.compressor is not None:
            return None
        sim = self._get_sim(path)  # raises KeyError for unknown path
        self._parse_chunk_key(path, chunk_key, sim)  # raises KeyError if out of bounds
        # Edge chunks are always padded to the full chunk shape before
        # serialisation, so the wire size is always prod(chunk_shape) * itemsize.
        return int(np.prod(self.chunk_shapes[path])) * np.dtype(sim.dtype).itemsize

    def read_chunk(self, path, chunk_key):
        sim = self._get_sim(path)
        chunk_shape = self.chunk_shapes[path]
        chunk_index = self._parse_chunk_key(path, chunk_key, sim)

        indexers = {}
        for dim, chunk_i, chunk_size, size in zip(
            sim.dims,
            chunk_index,
            chunk_shape,
            sim.shape,
        ):
            start = int(chunk_i * chunk_size)
            stop = min(start + int(chunk_size), int(size))
            indexers[dim] = slice(start, stop)

        chunk = np.asarray(sim.isel(indexers).data)
        chunk = self._pad_edge_chunk(chunk, chunk_shape, sim.dtype)
        chunk = np.ascontiguousarray(chunk)

        if self.compressor is not None:
            return self.compressor.encode(chunk)

        return chunk.tobytes(order="C")

    def _get_sim(self, path):
        if path not in self.paths:
            raise KeyError(path)
        return self.sims[self.paths.index(path)]

    def _parse_chunk_key(self, path, chunk_key, sim):
        parts = chunk_key.strip("/").split("/")
        if len(parts) != sim.ndim:
            raise KeyError(chunk_key)

        try:
            chunk_index = tuple(int(part) for part in parts)
        except ValueError as exc:
            raise KeyError(chunk_key) from exc

        chunk_shape = self.chunk_shapes[path]
        grid_shape = tuple(
            int(math.ceil(size / chunk))
            for size, chunk in zip(sim.shape, chunk_shape)
        )
        if any(
            index < 0 or index >= grid
            for index, grid in zip(chunk_index, grid_shape)
        ):
            raise KeyError(chunk_key)

        return chunk_index

    def _pad_edge_chunk(self, chunk, chunk_shape, dtype):
        if tuple(chunk.shape) == tuple(chunk_shape):
            return chunk.astype(dtype, copy=False)

        padded = np.full(
            chunk_shape,
            _fill_value_for_dtype(dtype),
            dtype=dtype,
        )
        insert = tuple(slice(0, size) for size in chunk.shape)
        padded[insert] = chunk
        return padded

    def _parse_data_key(self, key):
        """Split a chunk URL key into ``(scale_path, chunk_coords)``.

        Standard OME-Zarr chunk keys follow the pattern ``{scale}/{chunk}``.
        """
        parts = key.split("/", 1)
        if len(parts) != 2:
            raise KeyError(key)
        return parts[0], parts[1]


# ---------------------------------------------------------------------------
# HCS plate helpers
# ---------------------------------------------------------------------------

def _is_hcs_plate_tree(datatree):
    """Return True if *datatree* is an HCS plate tree rather than an msim.

    An msim has ``scale0``, ``scale1``, … as direct children; an HCS plate
    tree has row/column/FOV children instead.
    """
    return (
        isinstance(datatree, DataTree)
        and not msi_utils.get_sorted_scale_keys(datatree)
        and bool(datatree.children)
    )


class VirtualOMEZarrHCSPlate:
    """Read-only virtual OME-Zarr HCS plate backed by a DataTree of msims.

    *plate_tree* must contain msim sub-trees at paths of the form
    ``{row}/{column}/{fov}`` (e.g. ``"B/1/0"``).  Each FOV sub-tree is
    wrapped in a :class:`VirtualOMEZarr` and served under the corresponding
    HCS well path.  Only a single acquisition (id=0) is assumed.

    Parameters
    ----------
    plate_tree:
        DataTree with msim elements at ``{row}/{col}/{fov}`` positions.
    name:
        Plate name embedded in the OME-Zarr HCS metadata.
    compressor:
        Optional numcodecs compressor applied to every chunk.
    """

    def __init__(self, plate_tree, name="plate", compressor=None):
        self.name = name
        self.compressor = compressor

        # Walk 3 levels deep to find msim nodes at {row}/{col}/{fov}.
        self._fov_zarrs = {}  # (row, col, fov) -> VirtualOMEZarr

        for row, row_tree in plate_tree.children.items():
            for col, col_tree in row_tree.children.items():
                for fov, fov_tree in col_tree.children.items():
                    if msi_utils.get_sorted_scale_keys(fov_tree):
                        self._fov_zarrs[(row, col, fov)] = VirtualOMEZarr(
                            fov_tree, compressor=compressor
                        )

        if not self._fov_zarrs:
            raise ValueError(
                "plate_tree contains no msim FOVs at row/column/fov depth."
            )

        # Derive sorted row / column lists from the FOVs that were found.
        rows_set = {r for r, _, _ in self._fov_zarrs}
        cols_set = {c for _, c, _ in self._fov_zarrs}
        self._row_list = sorted(rows_set)
        self._col_list = sorted(
            cols_set, key=lambda x: int(x) if x.isdigit() else x
        )

        # (row, col) -> sorted list of fov keys present in that well.
        self._fov_map = {}
        for row, col, fov in self._fov_zarrs:
            self._fov_map.setdefault((row, col), []).append(fov)
        for wk in self._fov_map:
            self._fov_map[wk] = sorted(
                self._fov_map[wk], key=lambda x: int(x) if x.isdigit() else x
            )

        self._root_zattrs = self._build_root_zattrs()

    def _build_root_zattrs(self):
        """Build OME-Zarr HCS 0.4 plate-level metadata."""
        rows = [{"name": r} for r in self._row_list]
        columns = [{"name": c} for c in self._col_list]
        wells = [
            {
                "path": f"{row}/{col}",
                "rowIndex": self._row_list.index(row),
                "columnIndex": self._col_list.index(col),
            }
            for row in self._row_list
            for col in self._col_list
            if (row, col) in self._fov_map
        ]
        return {
            "plate": {
                "version": "0.4",
                "name": self.name,
                "acquisitions": [{"id": 0}],
                "columns": columns,
                "rows": rows,
                "wells": wells,
            }
        }

    def root_zgroup(self):
        return {"zarr_format": 2}

    def root_zattrs(self):
        return self._root_zattrs

    def _well_zattrs(self, row, col):
        """Build OME-Zarr HCS 0.4 well metadata for *row*/*col*."""
        fovs = self._fov_map.get((row, col), [])
        images = [{"path": fov, "acquisition": 0} for fov in fovs]
        return {"well": {"images": images, "version": "0.4"}}

    def consolidated_metadata(self):
        """Return consolidated Zarr metadata for the entire plate hierarchy."""
        metadata = {
            ".zgroup": self.root_zgroup(),
            ".zattrs": self.root_zattrs(),
        }
        for row in self._row_list:
            metadata[f"{row}/.zgroup"] = {"zarr_format": 2}
            for col in self._col_list:
                if (row, col) not in self._fov_map:
                    continue
                metadata[f"{row}/{col}/.zgroup"] = {"zarr_format": 2}
                metadata[f"{row}/{col}/.zattrs"] = self._well_zattrs(row, col)
                for fov in self._fov_map[(row, col)]:
                    fov_zarr = self._fov_zarrs[(row, col, fov)]
                    for sub_key, sub_val in (
                        fov_zarr.consolidated_metadata()["metadata"].items()
                    ):
                        full_key = f"{row}/{col}/{fov}/{sub_key}".rstrip("/")
                        metadata[full_key] = sub_val
        return {"zarr_consolidated_format": 1, "metadata": metadata}

    def get_json_key(self, key):
        """Route a metadata key request to the correct level of the hierarchy."""
        key = key.strip("/")

        if key == ".zgroup":
            return self.root_zgroup()
        if key == ".zattrs":
            return self.root_zattrs()
        if key == ".zmetadata":
            return self.consolidated_metadata()

        parts = key.split("/")

        # Row group: {row}/.zgroup
        if len(parts) == 2 and parts[1] == ".zgroup" and parts[0] in self._row_list:
            return {"zarr_format": 2}

        # Well level: {row}/{col}/.zgroup  or  {row}/{col}/.zattrs
        if len(parts) == 3 and parts[0] in self._row_list:
            row, col, meta = parts
            if (row, col) in self._fov_map:
                if meta == ".zgroup":
                    return {"zarr_format": 2}
                if meta == ".zattrs":
                    return self._well_zattrs(row, col)

        # FOV and deeper: {row}/{col}/{fov}/… — delegate to per-FOV VirtualOMEZarr.
        if len(parts) >= 4:
            row, col, fov = parts[0], parts[1], parts[2]
            fov_key = "/".join(parts[3:])
            fov_zarr = self._fov_zarrs.get((row, col, fov))
            if fov_zarr is not None:
                return fov_zarr.get_json_key(fov_key)

        raise KeyError(key)

    def _parse_data_key(self, key):
        """Split an HCS chunk URL key into ``(path, chunk_key)``.

        HCS chunk keys follow ``{row}/{col}/{fov}/{scale}/{chunk_coords}``.
        The first four components form the *path*; the remainder is the
        chunk coordinate string passed to :meth:`read_chunk`.
        """
        parts = key.strip("/").split("/")
        # Need row/col/fov/scale plus at least one chunk-coordinate component.
        if len(parts) < 5:
            raise KeyError(key)
        path = "/".join(parts[:4])
        chunk_key = "/".join(parts[4:])
        return path, chunk_key

    def read_chunk(self, path, chunk_key):
        """Read a chunk; *path* is ``{row}/{col}/{fov}/{scale}``."""
        parts = path.strip("/").split("/")
        if len(parts) != 4:
            raise KeyError(path)
        row, col, fov, scale = parts
        fov_zarr = self._fov_zarrs.get((row, col, fov))
        if fov_zarr is None:
            raise KeyError(path)
        return fov_zarr.read_chunk(scale, chunk_key)

    def chunk_content_length(self, path, chunk_key):
        """Return chunk byte length without materialising the dask graph."""
        parts = path.strip("/").split("/")
        if len(parts) != 4:
            raise KeyError(path)
        row, col, fov, scale = parts
        fov_zarr = self._fov_zarrs.get((row, col, fov))
        if fov_zarr is None:
            raise KeyError(path)
        return fov_zarr.chunk_content_length(scale, chunk_key)


# ---------------------------------------------------------------------------
# Module-level SIGINT routing
# When a VirtualOMEZarrServer is created from the main thread, its _stopped
# event is registered here.  A shared SIGINT handler sets every registered
# event so that serve_forever() unblocks even when it runs in a worker thread
# (e.g. a Jupyter / IPyKernel 6+ thread-pool cell), where signals are
# delivered only to the *main* thread.
# ---------------------------------------------------------------------------
_sigint_lock = threading.Lock()
_sigint_stop_events: list = []
_sigint_prev_handler = None


def _sigint_handler(sig, frame):
    with _sigint_lock:
        for ev in list(_sigint_stop_events):
            ev.set()
    prev = _sigint_prev_handler
    if callable(prev):
        prev(sig, frame)
    else:
        raise KeyboardInterrupt


def _register_sigint_stop_event(event):
    """Register *event* to be set on SIGINT.  No-op when not on main thread."""
    global _sigint_prev_handler
    if threading.current_thread() is not threading.main_thread():
        return
    try:
        with _sigint_lock:
            if event not in _sigint_stop_events:
                _sigint_stop_events.append(event)
            cur = signal.getsignal(signal.SIGINT)
            if cur is not _sigint_handler:
                _sigint_prev_handler = cur
                signal.signal(signal.SIGINT, _sigint_handler)
    except (ValueError, OSError):
        with _sigint_lock:
            if event in _sigint_stop_events:
                _sigint_stop_events.remove(event)


def _unregister_sigint_stop_event(event):
    """Remove *event* from the SIGINT registry; restore the original handler
    when the registry becomes empty."""
    global _sigint_prev_handler
    try:
        with _sigint_lock:
            if event in _sigint_stop_events:
                _sigint_stop_events.remove(event)
            if not _sigint_stop_events:
                cur = signal.getsignal(signal.SIGINT)
                if cur is _sigint_handler and _sigint_prev_handler is not None:
                    try:
                        signal.signal(signal.SIGINT, _sigint_prev_handler)
                    except (ValueError, OSError):
                        pass
                    _sigint_prev_handler = None
    except Exception:
        pass


class VirtualOMEZarrServer:
    def __init__(
        self,
        virtual_zarrs,
        host="127.0.0.1",
        port=8000,
        route_prefix="image",
        max_concurrent_chunks=None,
    ):
        # Build the routing dict. Each virtual zarr's name is derived from its
        # route key so the URL path and the embedded metadata name stay in sync.
        self.virtual_zarrs = {}
        for index, virtual_zarr in enumerate(virtual_zarrs):
            route_name = f"{route_prefix}_{index}.ome.zarr"
            virtual_zarr.name = route_name
            # Update the name embedded in root metadata (differs by zarr type).
            if isinstance(virtual_zarr, VirtualOMEZarr):
                virtual_zarr._root_zattrs["multiscales"][0]["name"] = route_name
            elif isinstance(virtual_zarr, VirtualOMEZarrHCSPlate):
                virtual_zarr._root_zattrs["plate"]["name"] = route_name
            self.virtual_zarrs[route_name] = virtual_zarr
        self.host = host
        self.port = int(port)
        self.max_concurrent_chunks = (
            max_concurrent_chunks
            if max_concurrent_chunks is not None
            else min(4, os.cpu_count() or 1)
        )
        # Unique token per server instance so URLs change on every restart,
        # preventing web viewers (e.g. Neuroglancer) from serving stale
        # in-memory cache when the same port is reused with different data.
        self._session_token = uuid.uuid4().hex[:8]
        self.urls = [
            f"http://{self.host}:{self.port}/{self._session_token}/{name}"
            for name in self.virtual_zarrs
        ]
        self._loop = None
        self._runner = None
        self._thread = None
        self._started = threading.Event()
        self._stopped = threading.Event()
        self._start_error = None
        # Register cleanup hooks as early as possible (while still on the
        # calling thread, which may be the main thread).
        atexit.register(self.stop)
        _register_sigint_stop_event(self._stopped)

    def serve_forever(self):
        self.start()
        # Also try here in case serve_forever() is called directly from the
        # main thread (e.g. in a plain script) and __init__ was not.
        _register_sigint_stop_event(self._stopped)
        try:
            # Use a bounded wait (0.5 s) instead of blocking indefinitely.
            # On POSIX the SIGINT handler sets _stopped, so the wait returns
            # almost immediately.  On Windows, IPyKernel injects
            # KeyboardInterrupt via PyThreadState_SetAsyncExc; Python only
            # checks async exceptions at bytecode boundaries, so a finite
            # timeout ensures the exception can fire within each polling cycle.
            while not self._stopped.wait(timeout=0.5):
                pass
        except KeyboardInterrupt:
            pass
        finally:
            _unregister_sigint_stop_event(self._stopped)
            self.stop()

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return

        self._started.clear()
        self._stopped.clear()
        self._start_error = None
        self._thread = threading.Thread(
            target=self._run_loop_thread,
            daemon=True,
        )
        self._thread.start()
        self._started.wait()
        if self._start_error is not None:
            raise self._start_error

    def stop(self):
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._stopped.set()
        _unregister_sigint_stop_event(self._stopped)
        atexit.unregister(self.stop)

    def _run_loop_thread(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._start_async())
            self._started.set()
            self._loop.run_forever()
        except BaseException as exc:
            self._start_error = exc
            self._started.set()
        finally:
            if self._runner is not None:
                self._loop.run_until_complete(self._runner.cleanup())
            self._loop.close()
            self._stopped.set()

    async def _start_async(self):
        from aiohttp import web

        app = web.Application()
        app["virtual_zarrs"] = self.virtual_zarrs
        app["chunk_semaphore"] = asyncio.Semaphore(
            self.max_concurrent_chunks
        )
        token = self._session_token
        app.router.add_route(
            "*", f"/{token}/{{image_name}}", _handle_virtual_zarr_request
        )
        app.router.add_route(
            "*",
            f"/{token}/{{image_name}}/{{key:.*}}",
            _handle_virtual_zarr_request,
        )

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        print(
            f"Serving virtual OME-Zarrs at http://{self.host}:{self.port} "
            "until interrupted..."
        )


async def _handle_virtual_zarr_request(request):
    from aiohttp import web

    cors_headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        # Private Network Access: allows public pages (e.g. the OME-Zarr
        # validator on ome.github.io) to fetch from localhost in Chrome.
        "Access-Control-Allow-Private-Network": "true",
    }

    if request.method == "OPTIONS":
        return web.Response(status=204, headers=cors_headers)

    if request.method not in {"GET", "HEAD"}:
        raise web.HTTPMethodNotAllowed(
            request.method,
            ["GET", "HEAD", "OPTIONS"],
            headers=cors_headers,
        )

    image_name = request.match_info["image_name"]
    key = request.match_info.get("key", "").strip("/")
    virtual_zarr = request.app["virtual_zarrs"].get(image_name)
    if virtual_zarr is None:
        raise web.HTTPNotFound(headers=cors_headers)

    if not key:
        if not request.path.endswith("/"):
            location = request.path + "/"
            if request.query_string:
                location += "?" + request.query_string
            raise web.HTTPPermanentRedirect(
                location=location,
                headers=cors_headers,
            )
        return web.Response(status=204, headers=cors_headers)

    try:
        json_obj = virtual_zarr.get_json_key(key)
    except KeyError:
        json_obj = None

    if json_obj is not None:
        payload = _json_response_dict(json_obj)
        return web.Response(
            body=b"" if request.method == "HEAD" else payload,
            content_type="application/json",
            headers={
                **cors_headers,
                "Content-Length": str(len(payload)),
                "Cache-Control": "no-store",
            },
        )

    # Delegate key splitting to each virtual zarr type: VirtualOMEZarr uses
    # "{scale}/{chunk}" while VirtualOMEZarrHCSPlate uses "{row}/{col}/{fov}/{scale}/{chunk}".
    try:
        path, chunk_key = virtual_zarr._parse_data_key(key)
    except KeyError:
        raise web.HTTPNotFound(headers=cors_headers)

    # Short-circuit HEAD requests: validate existence cheaply without
    # materialising the (potentially expensive) dask graph.
    if request.method == "HEAD":
        try:
            content_length = virtual_zarr.chunk_content_length(path, chunk_key)
        except KeyError:
            raise web.HTTPNotFound(headers=cors_headers)
        head_headers = {**cors_headers, "Cache-Control": "no-store"}
        if content_length is not None:
            head_headers["Content-Length"] = str(content_length)
        return web.Response(
            body=b"",
            content_type="application/octet-stream",
            headers=head_headers,
        )

    try:
        async with request.app["chunk_semaphore"]:
            payload = await asyncio.to_thread(
                virtual_zarr.read_chunk,
                path,
                chunk_key,
            )
    except KeyError:
        raise web.HTTPNotFound(headers=cors_headers)

    return web.Response(
        body=b"" if request.method == "HEAD" else payload,
        content_type="application/octet-stream",
        headers={
            **cors_headers,
            "Content-Length": str(len(payload)),
            "Cache-Control": "no-store",
        },
    )


def serve_virtual_ome_zarrs(
    msims,
    host="127.0.0.1",
    port=8000,
    route_prefix="image",
    max_concurrent_chunks=None,
    compressor=None,
    omero_channels=None,
):
    """Serve a list of msims or HCS plate DataTrees as virtual OME-Zarrs.

    Each element is auto-detected: DataTrees with ``scale0``/``scale1``
    children are served as OME-Zarr multiscale images; DataTrees with
    row/column/FOV depth are served as OME-Zarr HCS plates.
    Names are derived from the route keys inside :class:`VirtualOMEZarrServer`
    so the URL path and the embedded metadata name always stay in sync.

    ``omero_channels``, when provided, supplies temporary OMERO metadata for
    each image without modifying the input DataTrees.
    """
    if omero_channels is not None and len(omero_channels) != len(msims):
        raise ValueError("omero_channels must match the number of msims.")

    virtual_zarrs = []
    for index, element in enumerate(msims):
        omero = None if omero_channels is None else omero_channels[index]
        if _is_hcs_plate_tree(element):
            if omero is not None:
                raise ValueError(
                    "omero_channels are not supported for HCS plate trees."
                )
            virtual_zarrs.append(
                VirtualOMEZarrHCSPlate(element, compressor=compressor)
            )
        else:
            virtual_zarrs.append(
                VirtualOMEZarr(element, compressor=compressor, omero=omero)
            )
    return VirtualOMEZarrServer(
        virtual_zarrs,
        host=host,
        port=port,
        route_prefix=route_prefix,
        max_concurrent_chunks=max_concurrent_chunks,
    )


def sim_to_ngff_image(sim, transform_key):
    """Convert a SpatialImage (sim) to an ngff-zarr NgffImage.

    Spacing and origin become scale and translation. If transform_key is
    provided, its static translation is added to the origin. General affine
    transformations are rejected: NgffImage carries calibration, whereas
    additional OME-Zarr coordinate transformations belong to Multiscales
    metadata. Use msim_to_ngff_multiscales(..., ngff_version="0.6") for those.
    """

    sdims = si_utils.get_spatial_dims_from_sim(sim)
    origin = si_utils.get_origin_from_sim(sim)
    if transform_key is not None:
        transform = _ngff_v06.static_affine(sim, transform_key)
        if not np.allclose(transform[:-1, :-1], np.eye(len(sdims))):
            raise ValueError(
                "Scale/translation export cannot represent this registration; "
                "use msim_to_ngff_multiscales(..., ngff_version='0.6')."
            )
        transform_translation = param_utils.translation_from_affine(transform)
        for isdim, sdim in enumerate(sdims):
            origin[sdim] = origin[sdim] + transform_translation[isdim]

    time = get_ngff_time_transform(sim)
    scale = si_utils.get_spacing_from_sim(sim)
    if "t" in sim.dims:
        scale["t"] = time["scale"]
        origin["t"] = time["translation"]
    return ngff_zarr.to_ngff_image(
        sim.data,
        dims=sim.dims,
        scale=scale,
        translation=origin,
        axes_units=sim.attrs.get(NGFF_AXES_UNITS_ATTR),
    )


def msim_to_ngff_multiscales(
    msim,
    transform_key,
    ngff_version="0.4",
    target_coordinate_system="registered",
):
    """Convert an existing pyramid to NGFF metadata and lazy arrays.

    The default 0.4 representation folds a static translation into each
    level's origin. Version 0.6 keeps the full registration affine separate
    from per-level calibration and names its target coordinate system.
    """

    if ngff_version == "0.6":
        sims = [
            msi_utils.get_sim_from_msim(msim, scale=key)
            for key in msi_utils.get_sorted_scale_keys(msim)
        ]
        affines = [_ngff_v06.static_affine(sim, transform_key) for sim in sims]
        if any(not np.allclose(a, affines[0]) for a in affines[1:]):
            raise ValueError(
                "Registration must be identical across pyramid levels."
            )
        images = [sim_to_ngff_image(sim, transform_key=None) for sim in sims]
        datasets = []
        for level, sim in enumerate(sims):
            sdims = si_utils.get_spatial_dims_from_sim(sim)
            coordtfs, axes = calc_ngff_coordinate_transformations_and_axes(
                {
                    "spacing": si_utils.get_spacing_from_sim(sim),
                    "origin": si_utils.get_origin_from_sim(sim),
                },
                [{d: 1 for d in sdims}],
                nsdims=si_utils.get_nonspatial_dims_from_sim(sim),
                time_transform=get_ngff_time_transform(sim),
            )
            for axis in axes:
                units = sim.attrs.get(NGFF_AXES_UNITS_ATTR, {})
                if axis["name"] in units:
                    axis["unit"] = units[axis["name"]]
            datasets.append(
                {
                    "path": f"scale{level}/image",
                    "coordinateTransformations": coordtfs[0],
                }
            )
        metadata = _ngff_v06.build_metadata(
            _drop_none_values(axes),
            datasets,
            "image",
            affines[0] if transform_key is not None else None,
            target_coordinate_system,
        )
        return ngff_zarr.Multiscales(
            images, metadata=metadata, scale_factors=[]
        )
    if ngff_version != "0.4":
        raise ValueError("In-memory export supports versions '0.4' and '0.6'.")

    ngff_ims = []
    for scale_key in msi_utils.get_sorted_scale_keys(msim):
        sim = msi_utils.get_sim_from_msim(msim, scale=scale_key)
        ngff_ims.append(sim_to_ngff_image(sim, transform_key=transform_key))

    # workaround for creating multiscale metadata
    # does this create significant overhead?
    ngff_multiscales_scales = [
        ngff_zarr.to_multiscales(ngff_im, scale_factors=[])
        for ngff_im in ngff_ims
    ]

    sdims = msi_utils.get_spatial_dims(msim)

    ngff_multiscales_scales_v04 = [
        ms.metadata.to_version("0.4") for ms in ngff_multiscales_scales
    ]

    ngff_multiscales = ngff_zarr.Multiscales(
        ngff_ims,
        metadata=ngff_zarr.Metadata(
            axes=ngff_multiscales_scales_v04[0].axes,
            datasets=[
                ngff_zarr.Dataset(
                    path="scale%s/image" % iscale,
                    coordinateTransformations=ngff_multiscales_scale_v04.datasets[
                        0
                    ].coordinateTransformations,
                )
                for iscale, ngff_multiscales_scale_v04 in enumerate(
                    ngff_multiscales_scales_v04
                )
            ],
            coordinateTransformations=None,
        ),
        scale_factors=[
            {
                sdim: int(
                    ngff_ims[0].data.shape[ngff_ims[0].dims.index(sdim)]
                    / ngff_ims[iscale].data.shape[
                        ngff_ims[iscale].dims.index(sdim)
                    ]
                )
                for sdim in sdims
            }
            for iscale in range(1, len(ngff_ims))
        ],
    )

    return ngff_multiscales


def ngff_image_to_sim(ngff_im, transform_key, data=None, affine=None):
    """Convert an ngff-zarr NgffImage to a SpatialImage (xarray.DataArray).

    Scale and translation define the sim's spatial coordinates. The optional
    homogeneous spatial affine maps these coordinates to a target coordinate
    system and is stored under transform_key; None stores identity. Additional
    OME-Zarr coordinate transformations must be resolved by the caller.
    """

    # Reuse the general sim constructor so zarr-backed reads preserve chunk
    # hints and singleton t/c axes lazily.
    sim = si_utils.get_sim_from_array(
        ngff_im.data if data is None else data,
        dims=ngff_im.dims,
        scale=ngff_im.scale,
        translation=ngff_im.translation,
        transform_key=transform_key,
    )

    # Keep the distinction between axes stored in NGFF and singleton axes
    # added by get_sim_from_array().  Consumers that address the original
    # OME-Zarr (not a newly generated virtual one) need the former.
    sim.attrs[NGFF_SOURCE_DIMS_ATTR] = list(ngff_im.dims)
    sim.attrs[NGFF_AXES_UNITS_ATTR] = dict(ngff_im.axes_units or {})

    # get_sim_from_array() applies scale and translation to the spatial
    # coordinates only; ``t`` coordinates stay frame indices.  Keep the time
    # calibration alongside them so it can be written back out and reported to
    # viewers reading the original store.
    if "t" in ngff_im.dims:
        set_ngff_time_transform(sim, _ngff_time_transform_of(ngff_im))

    sdims = si_utils.get_spatial_dims_from_sim(sim)

    si_utils.set_sim_affine(
        sim,
        param_utils.affine_to_xaffine(
            np.eye(len(sdims) + 1) if affine is None else affine,
            t_coords=sim.coords["t"].values,
        ),
        transform_key=transform_key,
    )

    return sim


def ngff_multiscales_to_msim(
    ngff_multiscales,
    transform_key,
    data_arrays=None,
    target_coordinate_system=None,
):
    """Convert ngff-zarr Multiscales to a MultiscaleSpatialImage (DataTree).

    Each resolution level retains its scale and translation. A supported
    additional coordinate transformation is resolved to a spatial affine and
    stored under transform_key. target_coordinate_system selects the OME-Zarr
    coordinate system; it need not have the same name as transform_key.
    """

    if data_arrays is None:
        data_arrays = [None] * len(ngff_multiscales.images)

    affine = _registration_from_ngff(
        ngff_multiscales, target_coordinate_system
    )
    msim_dict = {}
    for iscale, (ngff_im, data_array) in enumerate(
        zip(ngff_multiscales.images, data_arrays)
    ):
        sim = ngff_image_to_sim(
            ngff_im,
            transform_key=transform_key,
            data=data_array,
            affine=affine,
        )
        curr_scale_msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
        msim_dict[f"scale{iscale}"] = curr_scale_msim["scale0"]

    return DataTree.from_dict(msim_dict)


def _open_ngff_dataset_arrays(zarr_path, ngff_multiscales):
    # ngff_zarr currently reads image data as dask arrays. For the zarr-backed
    # default path, reuse its parsed metadata but reopen the on-disk arrays.
    return [
        _zarr_compat.open_zarr_array(zarr_path, dataset.path, mode="r")
        for dataset in ngff_multiscales.metadata.datasets
    ]


def read_ngff_multiscales(zarr_path):
    """Parse OME-Zarr 0.4/0.5 or supported 0.6 draft metadata lazily.

    ``zarr_path`` may be a path/URL or an already-constructed zarr store, which
    is how the browser runtime routes reads through its service worker. Parsing
    reads metadata only - no chunk is fetched until the arrays are used.
    """
    root = _zarr_compat.open_zarr_group(zarr_path, mode="r")
    version = root.attrs.get("ome", {}).get("version", "")
    if version.startswith("0.6") and version not in ("0.6", "0.6.dev4"):
        raise ValueError(
            f"Unsupported OME-Zarr revision {version!r}; "
            "this adapter targets the ngff-zarr 0.43 '0.6' model."
        )
    multiscales = ngff_zarr.from_ngff_zarr(zarr_path)
    if version in ("0.6", "0.6.dev4"):
        _ngff_v06.validate_calibration(multiscales.metadata)
        # ngff-zarr 0.43 derives dims from the first system, which need not be
        # intrinsic. Do not silently interpret such data in a different basis.
        intrinsic = multiscales.metadata.intrinsic_coordinate_system
        if list(multiscales.images[0].dims) != [
            a.name for a in intrinsic.axes
        ]:
            raise NotImplementedError("Coordinate system axis orders differ.")
    return multiscales


def _registration_from_ngff(multiscales, target_coordinate_system):
    metadata = multiscales.metadata
    if hasattr(metadata, "coordinateSystems"):
        _ngff_v06.validate_calibration(metadata)
        return _ngff_v06.spatial_affine(metadata, target_coordinate_system)
    if target_coordinate_system is not None:
        raise ValueError("This metadata has no named coordinate systems.")
    return None


def write_multiscales_metadata(
    group,
    axes,
    datasets,
    ngff_version="0.4",
    affine=None,
    target_coordinate_system="registered",
):
    """Write OME-Zarr multiscales metadata without writing image arrays.

    Version-specific ngff-zarr metadata classes describe the document. OME-Zarr
    0.4 stores multiscales at the top level of group attributes. OME-Zarr 0.5
    and the supported 0.6 draft use the ome namespace.

    ngff_version="0.6" writes the 0.6.dev4 schema revision: dataset coordinate
    transformations map array coordinates to the intrinsic coordinate system.
    If affine is given, a multiscales-level coordinate transformation maps
    intrinsic coordinates to target_coordinate_system.
    """
    zarr_group_creation_kwargs_for_ngff_version(ngff_version)
    if ngff_version == "0.6":
        metadata = _ngff_v06.build_metadata(
            axes, datasets, group.name, affine, target_coordinate_system
        )
        multiscale = _drop_none_values(asdict(metadata))
        for key in ("extra", "omero"):
            multiscale.pop(key, None)
        ome = dict(group.attrs.get("ome", {}))
        ome.update(version="0.6.dev4", multiscales=[multiscale])
        group.attrs["ome"] = ome
        return
    if affine is not None:
        raise ValueError("Registration export requires ngff_version='0.6'.")
    metadata = ngff_zarr.Metadata(
        axes=[ngff_zarr.Axis(**dict(axis)) for axis in axes],
        datasets=[
            ngff_zarr.Dataset(
                path=dataset["path"],
                coordinateTransformations=[
                    _ngff_transform(transform)
                    for transform in dataset["coordinateTransformations"]
                ],
            )
            for dataset in datasets
        ],
        coordinateTransformations=None,
        name=group.name,
    )

    multiscale = _drop_none_values(asdict(metadata))
    # Only `axes`, `datasets` and `name` are written. The optional keys are
    # dropped rather than emitted empty, which keeps the document identical to
    # what the previous two writers produced.
    for optional in ("coordinateTransformations", "omero", "metadata",
                     "extra", "type", "version"):
        multiscale.pop(optional, None)

    if str(ngff_version).startswith("0.4"):
        multiscale["version"] = str(ngff_version)
        group.attrs["multiscales"] = [multiscale]
        return

    ome = dict(group.attrs.get("ome", {}))
    ome["version"] = str(ngff_version)
    ome["multiscales"] = [multiscale]
    group.attrs["ome"] = ome


def _ngff_transform(transform):
    """One coordinate transformation as an ngff-zarr dataclass."""
    transform = dict(transform)
    if transform.get("type") == "scale":
        return ngff_zarr.Scale(scale=list(transform["scale"]))
    if transform.get("type") == "translation":
        return ngff_zarr.Translation(translation=list(transform["translation"]))
    if transform.get("type") == "identity":
        return ngff_zarr.Identity()
    raise ValueError(f"Unsupported NGFF transform: {transform.get('type')!r}")


def zarr_group_creation_kwargs_for_ngff_version(ngff_version):
    """Select the Zarr storage format for the requested OME-Zarr version.

    OME-Zarr 0.4 uses Zarr format 2. OME-Zarr 0.5 and the supported 0.6 draft
    use Zarr format 3. The package requires zarr-python 3 or later.
    """
    if str(ngff_version).startswith("0.4"):
        if zarr.__version__ >= "3":
            return {"zarr_format": 2}
        return {}
    if ngff_version in ("0.5", "0.6"):
        return {"zarr_format": 3}
    raise ValueError(f"ngff_version {ngff_version} not supported")


def update_zarr_array_creation_kwargs_for_ngff_version(
    ngff_version, zarr_array_creation_kwargs):

    if zarr_array_creation_kwargs is None:
        zarr_array_creation_kwargs = {}
    if ngff_version == "0.4":
        zarr_array_creation_kwargs.update({
                "dimension_separator": '/',
        })
        if zarr.__version__ >= "3":
            zarr_array_creation_kwargs.update({
                "zarr_format": 2,
            })
    elif ngff_version in ("0.5", "0.6"):
        if zarr.__version__ < "3":
            raise ValueError("zarr>=3 required for ngff_version 0.5")
        zarr_array_creation_kwargs.update({
                "zarr_version" if zarr.__version__ < "3"
                else "zarr_format": 3,
        })
    else:
        raise ValueError(f"ngff_version {ngff_version} not supported")
    return zarr_array_creation_kwargs


# thanks to https://github.com/CamachoDejay/teaching-bioimage-analysis-python/blob/6076e00e392075ba9c07e67e868a39d4889e6298/short_examples/zarr-from-tiles/zarr-minimal-example-tiles.ipynb
def mean_dtype(arr, **kwargs):
    return np.mean(arr, **kwargs).astype(arr.dtype)


def write_and_return_downsampled_sim(
    array,
    dims: list[str],
    output_zarr_array_url: str,
    chunksizes: list[int],
    downscale_factors_per_spatial_dim: dict[str, int] = None,
    overwrite: bool = False,
    zarr_array_creation_kwargs: dict = None,
    res_level: int = 0,
    show_progressbar: bool = True,
    n_batch=1,
    batch_func=None,
    batch_func_kwargs=None,
):

    sdims = [dim for dim in dims if dim in si_utils.SPATIAL_DIMS]

    if not overwrite and os.path.exists(output_zarr_array_url):
        print(f"Found existing resolution level {res_level}...")
        array = da.from_zarr(output_zarr_array_url)
    else:
        print(f"Writing resolution level {res_level}...")
        # use pure dask
        if n_batch is None:
            #downscale
            if downscale_factors_per_spatial_dim is not None\
                and np.max(list(downscale_factors_per_spatial_dim.values())) > 1:
                array = da.coarsen(
                    mean_dtype,
                    array,
                    axes={
                        idim: downscale_factors_per_spatial_dim[dim] if dim in sdims else 1
                        for idim, dim in enumerate(dims)
                    },
                    trim_excess=True,
                )

            # Open output array. This allows setting `write_empty_chunks=True`,
            # which cannot be passed to dask.array.to_zarr below.
            output_zarr_arr = zarr.open(
                output_zarr_array_url,
                shape=array.shape,
                chunks=chunksizes,
                dtype=array.dtype,
                config={'write_empty_chunks': True},
                fill_value=0,
                mode="w",
                **zarr_array_creation_kwargs,
            )

            if show_progressbar:
                with dask.diagnostics.ProgressBar(show_progressbar): 
                    # Write the array
                    array = array.to_zarr(
                        output_zarr_arr,
                        overwrite=True,
                        return_stored=True,
                        compute=True,
                    )

            else:
                # Write the array
                array = array.to_zarr(
                    output_zarr_arr,
                    overwrite=True,
                    return_stored=True,
                    compute=True,
                )
        else:
            # use dask with batching to limit memory usage

            output_shape = [np.floor(s) // (downscale_factors_per_spatial_dim[sdim]
                    if sdim in sdims else 1)
                    for s, sdim in zip(array.shape, dims)]
            
            # make sure output array exists with correct shape and chunks, and with `write_empty_chunks=True`
            zarr.open(
                output_zarr_array_url,
                shape=[int(s) for s in output_shape],
                chunks=[int(cs) for cs in chunksizes],
                dtype=array.dtype,
                config={'write_empty_chunks': True},
                fill_value=0,
                mode="w" if overwrite else "a",
                **zarr_array_creation_kwargs,
            )

            write_downsampled_chunk_p = partial(write_downsampled_chunk, 
                input_array=array,
                output_shape=output_shape,
                dims=dims,
                output_zarr_array_url=output_zarr_array_url,
                output_chunksizes=chunksizes,
                downscale_factors_per_spatial_dim=downscale_factors_per_spatial_dim,
                zarr_array_creation_kwargs=zarr_array_creation_kwargs,
            )

            normalized_chunks = normalize_chunks(
                shape=output_shape,
                chunks=chunksizes,
            )

            nblocks = [len(nc) for nc in normalized_chunks]

            for batch in tqdm(
                misc_utils.ndindex_batches(nblocks, n_batch),
                total=int(np.ceil(np.prod(nblocks)/n_batch)))\
            if show_progressbar else\
                misc_utils.ndindex_batches(nblocks, n_batch):
                
                if batch_func is None:
                    for block_id in batch:
                        write_downsampled_chunk_p(block_id)
                else:
                    batch_func(
                        write_downsampled_chunk_p, batch,
                        **(batch_func_kwargs or {}))
                    
            array = da.from_zarr(output_zarr_array_url)
    return array


from dask.array.core import normalize_chunks
def write_downsampled_chunk(
    block_id,
    input_array,
    output_shape,
    output_chunksizes,
    dims,
    output_zarr_array_url,
    downscale_factors_per_spatial_dim,
    zarr_array_creation_kwargs,
):

    sdims = [dim for dim in dims if dim in si_utils.SPATIAL_DIMS]
    nsdims = [dim for dim in dims if dim not in si_utils.SPATIAL_DIMS]

    normalized_chunks = normalize_chunks(
        shape=output_shape,
        chunks=output_chunksizes,
    )

    ns_coord = {dim: block_id[idim] for idim, dim in enumerate(nsdims)}
    spatial_chunk_ind = block_id[len(nsdims):]

    chunk_offset = {
        sdims[idim]: int(np.sum(normalized_chunks[len(nsdims) + idim][:b]))
        if b > 0 else 0 for idim, b in enumerate(spatial_chunk_ind)}
    chunk_shape = {
        sdims[idim]: normalized_chunks[len(nsdims) + idim][b]
            for idim, b in enumerate(spatial_chunk_ind)}
    
    input_slices = tuple(
        slice(
            ns_coord[dim],
            ns_coord[dim] + 1,
        )
        if dim in nsdims
        else slice(
            chunk_offset[dim] * (downscale_factors_per_spatial_dim[dim]
                if dim in downscale_factors_per_spatial_dim else 1),
            (chunk_offset[dim] + chunk_shape[dim])
                * (downscale_factors_per_spatial_dim[dim]
                if dim in downscale_factors_per_spatial_dim else 1),
        )
        for dim in dims
    )

    output_chunk = da.coarsen(
        mean_dtype,
        input_array[input_slices],
        axes={
            idim: downscale_factors_per_spatial_dim[dim] if dim in sdims else 1
            for idim, dim in enumerate(dims)
        },
        trim_excess=True,
    )

    output_zarr_arr = zarr.open(
        output_zarr_array_url,
        shape=[int(s) for s in output_shape],
        chunks=[int(cs) for cs in output_chunksizes],
        dtype=input_array.dtype,
        config={'write_empty_chunks': True},
        fill_value=0,
        mode="a",
        **zarr_array_creation_kwargs,
    )

    output_zarr_arr[tuple(
        slice(
            ns_coord[dim],
            ns_coord[dim] + 1,
        )
        if dim in nsdims
        else slice(
            chunk_offset[dim],
            chunk_offset[dim] + chunk_shape[dim],
        )
        for dim in dims
    )] = output_chunk.compute()

    return


def calc_ngff_coordinate_transformations_and_axes(
    stack_properties_res0: dict,
    res_abs_factors: list[dict],
    nsdims: list = None,
    time_transform: dict = None,
):

    spacing = stack_properties_res0['spacing']
    origin = stack_properties_res0['origin']
    sdims = list(spacing.keys())
    n_resolutions = len(res_abs_factors)

    # Resolution levels differ spatially only, so the time calibration - which
    # the caller carries over from the images it derived this stack from -
    # applies unchanged to every level.
    time_transform = {
        **DEFAULT_NGFF_TIME_TRANSFORM,
        **(time_transform or {}),
    }
    nsdim_scales = [
        float(time_transform["scale"]) if dim == "t" else 1.0
        for dim in nsdims
    ]
    nsdim_translations = [
        float(time_transform["translation"]) if dim == "t" else 0
        for dim in nsdims
    ]

    coordtfs = [
            [
                {
                    "type": "scale",
                    "scale": nsdim_scales
                    + [
                        float(s * res_abs_factors[res_level][dim])
                        for dim, s in spacing.items()
                    ],
                },
                {
                    "type": "translation",
                    "translation": nsdim_translations
                    + [
                        origin[dim]
                        + (res_abs_factors[res_level][dim] - 1) * spacing[dim] / 2
                        for dim in sdims
                    ],
                },
            ]
            # [0] * (ndim - len(sdims)) + [origin[dim] for dim in sdims]}]
            for res_level in range(n_resolutions)
        ]

    axes = [
        {
            "name": dim,
            "type": "channel"
            if dim == "c"
            else ("time" if dim == "t" else "space"),
        }
        | ({"unit": "micrometer"} if dim in sdims else {})
        | (
            {"unit": time_transform["unit"]}
            if dim == "t" and time_transform["unit"]
            else {}
        )
        for dim in nsdims + sdims
    ]

    return coordtfs, axes


def write_sim_to_ome_zarr(
    sim,
    output_zarr_url: str,
    downscale_factors_per_spatial_dim: dict[str, int] = None,
    overwrite: bool = False,
    ngff_version: str = "0.4",
    zarr_array_creation_kwargs: dict = None,
    show_progressbar: bool = True,
    batch_options: dict | None = None,
    transform_key: str | None = None,
    target_coordinate_system: str = "registered",
):
    """
    Write a SpatialImage (sim) as an OME-Zarr multiscale image.

    Supports 0.4, 0.5 and the 0.6.dev4 draft via ngff_version="0.6".
    Returns the input sim.

    If overwrite is False, image data will be read from the zarr file
    and missing pyramid levels will be completed. OME-Zarr metadata
    will be overwritten in any case.

    With ngff_version="0.6", transform_key selects a static physical-space
    affine to store separately from voxel calibration. None writes calibration
    only (the historical behavior). The input sim is returned.

    Parameters
    ----------
    sim : xarray.DataArray
        SpatialImage to write
    output_zarr_url : str
        Path to the output zarr file
    downscale_factors_per_spatial_dim : dict, optional
        Downscale factors per spatial dimension to use for
        generating the resolution levels, by default None (automatic factors of 2 where the level size allows)
    overwrite : bool, optional
        Whether to overwrite existing data in the output zarr file,
        by default False
    ngff_version : str, optional
        OME-Zarr version selector: "0.4" (default), "0.5", or "0.6".
        "0.6" writes the 0.6.dev4 draft schema, not the 0.6rc0 release candidate.
    zarr_array_creation_kwargs : dict, optional
        Additional keyword arguments to pass to zarr.open
        when creating the zarr arrays, by default None
    show_progressbar : bool, optional
        Whether to show a progress bar (tqdm),
    transform_key : str or None, optional
        Package key selecting a static spatial affine; requires ngff_version="0.6".
        None writes only voxel calibration.
    target_coordinate_system : str, optional
        OME-Zarr output coordinate-system name, by default "registered".
        This name is independent of the package transform_key.
    batch_options : dict, optional
        Options for processing chunks in independent batches. Keys:
        - batch_func: Callable, optional
            Function to process each batch of chunks. Inputs:
            1) a list of block_id(s)
            2) function that performs fusion when passed a given block_id
            By default None, in which case the each block is processed sequentially.
        - n_batch: int
            Number of blocks to process in each batch.
            (n_batch>1 only compatible with a provided batch_func). By default 1.
        - batch_func_kwargs: dict, optional
            Additional keyword arguments passed to batch_func.

    """

    if batch_options is None:
        batch_options = {}

    n_batch = batch_options.get("n_batch", 1)
    batch_func = batch_options.get("batch_func", None)
    batch_func_kwargs = batch_options.get("batch_func_kwargs", None)

    if zarr_array_creation_kwargs is None:
        zarr_array_creation_kwargs = {}

    zarr_array_creation_kwargs = \
        update_zarr_array_creation_kwargs_for_ngff_version(
            ngff_version, zarr_array_creation_kwargs)

    zarr_group_creation_kwargs = zarr_group_creation_kwargs_for_ngff_version(
        ngff_version
    )

    dims = sim.dims
    nsdims = si_utils.get_nonspatial_dims_from_sim(sim)
    sdims = si_utils.get_spatial_dims_from_sim(sim)
    spacing = si_utils.get_spacing_from_sim(sim)
    origin = si_utils.get_origin_from_sim(sim)
    spatial_shape = {
        dim: sim.data.shape[idim]
        for idim, dim in enumerate(dims)
        if dim in sdims
    }

    res_shapes, res_rel_factors, res_abs_factors = \
        msi_utils.calc_resolution_levels(
            spatial_shape,
            downscale_factors_per_spatial_dim=downscale_factors_per_spatial_dim,
        )

    n_resolutions = len(res_shapes)

    coordtfs, axes = calc_ngff_coordinate_transformations_and_axes(
        {
            'spacing': spacing,
            'origin': origin,
            'shape': spatial_shape
        },
        res_abs_factors,
        nsdims=nsdims,
        time_transform=get_ngff_time_transform(sim),
    )

    affine = None
    if transform_key is not None:
        if ngff_version != "0.6":
            raise ValueError(
                "Registration export requires ngff_version='0.6'."
            )
        affine = _ngff_v06.static_affine(sim, transform_key)
    for axis in axes:
        units = sim.attrs.get(NGFF_AXES_UNITS_ATTR, {})
        if axis["name"] in units:
            axis["unit"] = units[axis["name"]]
    axes = _drop_none_values(axes)
    multiscales_datasets = [
        {"path": str(level), "coordinateTransformations": coordtfs[level]}
        for level in range(n_resolutions)
    ]
    if ngff_version == "0.6":
        zarr_array_creation_kwargs["dimension_names"] = list(dims)
        _ngff_v06.build_metadata(
            axes,
            multiscales_datasets,
            "image",
            affine,
            target_coordinate_system,
        )
    if overwrite and os.path.exists(output_zarr_url):
        shutil.rmtree(output_zarr_url)

    # parent_res_array = sim.data
    curr_res_array = sim.data  # in case of only one resolution level
    for res_level in range(0, n_resolutions):

        curr_res_array = write_and_return_downsampled_sim(
            curr_res_array,
            dims=dims,
            chunksizes=_chunk_shape_from_sim(sim),
            output_zarr_array_url=f"{output_zarr_url}/{res_level}",
            downscale_factors_per_spatial_dim=res_rel_factors[res_level],
            overwrite=overwrite,
            zarr_array_creation_kwargs=zarr_array_creation_kwargs,
            res_level=res_level,
            show_progressbar=show_progressbar,
            n_batch=n_batch,
            batch_func=batch_func,
            batch_func_kwargs=batch_func_kwargs,
        )

    output_group = zarr.open_group(
        output_zarr_url, mode="a", **zarr_group_creation_kwargs
    )

    write_multiscales_metadata(
        output_group,
        axes=axes,
        datasets=multiscales_datasets,
        ngff_version=ngff_version,
        affine=affine,
        target_coordinate_system=target_coordinate_system,
    )

    if "c" in sim.dims:
        contrast_min = np.array(
            curr_res_array.min(
                axis=[
                    idim for idim, dim in enumerate(sim.dims) if dim != "c"
                ]
            )
        )
        contrast_max = np.array(
            curr_res_array.max(
                axis=[
                    idim for idim, dim in enumerate(sim.dims) if dim != "c"
                ]
            )
        )

        omero = {
            "channels": [
                {
                    "color": "ffffff",
                    "label": f"{ch}",
                    "active": True,
                    "window": {
                        "end": int(contrast_max[ich]),
                        "max": int(contrast_max[ich]),
                        "min": 0,
                        "start": int(contrast_min[ich]),
                    },
                }
                for ich, ch in enumerate(sim.coords["c"].values)
            ],
        }
        if ngff_version == "0.4":
            output_group.attrs["omero"] = omero
        else:
            ome = dict(output_group.attrs["ome"])
            ome["omero"] = omero
            output_group.attrs["ome"] = ome

    return sim


def read_sim_from_ome_zarr(
    zarr_path,
    resolution_level=0,
    transform_key=si_utils.DEFAULT_TRANSFORM_KEY,
    array_backend="zarr",
    target_coordinate_system=None,
):
    """Read one resolution level as a SpatialImage (xarray.DataArray).

    Supports OME-Zarr 0.4/0.5 and the 0.6.dev4 model, including its "0.6"
    version alias. In 0.4/0.5, scale and translation define sim coordinates;
    the separate affine under transform_key is identity. The 0.6 adapter can
    also import an additional coordinate transformation as a spatial affine.

    Parameters
    ----------
    zarr_path : str, Path or zarr store
        Image group to read. Parent scene groups are not traversed.
    resolution_level : int, optional
        Resolution level to read; 0 is the highest resolution.
    transform_key : str, optional
        Package key under which to store the affine; defaults to
        si_utils.DEFAULT_TRANSFORM_KEY ("affine_metadata").
    array_backend : {"zarr", "dask"}, optional
        Lazy array backend, by default "zarr".
    target_coordinate_system : str or None, optional
        OME-Zarr coordinate-system name, independent of transform_key. None
        selects the sole supported additional transformation, or identity if
        none is present. Multiple candidates require explicit selection.
        Selecting the intrinsic coordinate system retains calibration only.

    Returns
    -------
    xarray.DataArray
        SpatialImage (sim) with the affine stored under transform_key.
    """
    if array_backend not in ("dask", "zarr"):
        raise ValueError("array_backend must be 'dask' or 'zarr'.")

    ngff_multiscales = read_ngff_multiscales(zarr_path)

    if resolution_level >= len(ngff_multiscales.images):
        raise ValueError(
            f"Resolution level {resolution_level} not found in {zarr_path}"
        )

    data = None
    if array_backend == "zarr":
        data = _open_ngff_dataset_arrays(zarr_path, ngff_multiscales)[
            resolution_level
        ]

    sim = ngff_image_to_sim(
        ngff_multiscales.images[resolution_level],
        transform_key=transform_key,
        data=data,
        affine=_registration_from_ngff(
            ngff_multiscales, target_coordinate_system
        ),
    )

    # get channel names from omero metadata if available
    root = _zarr_compat.open_zarr_group(zarr_path, mode="r")

    omero = root.attrs.get("ome", {}).get("omero", root.attrs.get("omero"))
    if omero is not None and "c" in sim.dims:
        ch_coords = [ch["label"] for ch in omero["channels"]]
        sim = sim.assign_coords(c=ch_coords)

    return sim


def _update_v06_registration(root, zarr_path, msim, transform_key, target):
    """Change only a registration edge, preserving calibration and pixels."""
    multiscales = read_ngff_multiscales(zarr_path)
    metadata = multiscales.metadata
    keys = msi_utils.get_sorted_scale_keys(msim)
    if len(keys) != len(metadata.datasets):
        raise ValueError(
            "Number of resolution levels does not match on-disk data."
        )
    intrinsic = metadata.intrinsic_coordinate_system
    axes = _drop_none_values([asdict(a) for a in intrinsic.axes])
    dims = [a.name for a in intrinsic.axes]
    affines = []
    for key, image in zip(keys, multiscales.images):
        sim = msi_utils.get_sim_from_msim(msim, scale=key)
        # Updating registration must not silently change the underlying grid.
        for dim in si_utils.get_spatial_dims_from_sim(sim):
            if not np.isclose(
                si_utils.get_spacing_from_sim(sim)[dim], image.scale[dim]
            ) or not np.isclose(
                si_utils.get_origin_from_sim(sim)[dim], image.translation[dim]
            ):
                raise ValueError("Calibration differs from the on-disk image.")
        if any(sim.sizes[d] != image.data.shape[dims.index(d)] for d in dims):
            raise ValueError("Image shape differs from the on-disk image.")
        affines.append(_ngff_v06.static_affine(sim, transform_key))
    if any(not np.allclose(a, affines[0]) for a in affines[1:]):
        raise ValueError(
            "Registration must be identical across pyramid levels."
        )
    edge = _drop_none_values(
        asdict(
            _ngff_v06.registration_transform(
                axes, affines[0], intrinsic.name, target
            )
        )
    )
    ome = deepcopy(dict(root.attrs["ome"]))
    entry = ome["multiscales"][0]
    systems = entry["coordinateSystems"]
    existing = [cs for cs in systems if cs["name"] == target]
    if existing and existing[0]["axes"] != axes:
        raise ValueError(
            "Target coordinate system axes differ from intrinsic."
        )
    if not existing:
        systems.append({"name": target, "axes": deepcopy(axes)})

    def is_replaced(tf):
        endpoints = (tf.get("input", {}), tf.get("output", {}))
        return not any(e.get("path") for e in endpoints) and {
            e.get("name") for e in endpoints
        } == {intrinsic.name, target}

    entry["coordinateTransformations"] = [
        tf
        for tf in entry.get("coordinateTransformations", [])
        if not is_replaced(tf)
    ] + [edge]
    root.attrs["ome"] = ome


def update_ome_zarr_multiscales_metadata(
    zarr_path,
    msim,
    transform_key,
    target_coordinate_system="registered",
):
    """Update image coordinate transformations without writing image arrays.

    For OME-Zarr 0.4/0.5, update each resolution level's scale and translation
    from msim; a selected static translation is added to the origin.
    For the supported 0.6 draft, update only the additional transformation
    between the intrinsic and target coordinate systems. Dataset calibration
    is preserved, and differing spatial grids are rejected.

    Other group metadata, including omero display settings, is preserved.

    Parameters
    ----------
    zarr_path : str, Path or zarr store
        Image group to update.
    msim : xarray.DataTree
        MultiscaleSpatialImage with the same resolution levels as the image.
    transform_key : str or None
        Package key selecting the affine. In 0.4/0.5 it must be a static
        translation; None uses the sim origin alone. In the 0.6 adapter it may
        be a full static spatial affine; None writes an identity transformation.
    target_coordinate_system : str, optional
        Output coordinate-system name for the 0.6 transformation, by default
        "registered". Independent of transform_key; unused for 0.4/0.5.

    Raises
    ------
    ValueError
        If the version, resolution levels or spatial grid are incompatible.
    NotImplementedError
        If the selected transformation is outside the supported subset.
    """
    root = _zarr_compat.open_zarr_group(zarr_path, mode="a")
    attrs = dict(root.attrs)

    if attrs.get("ome", {}).get("version") in ("0.6", "0.6.dev4"):
        _update_v06_registration(
            root, zarr_path, msim, transform_key, target_coordinate_system
        )
        return

    # Detect OME-Zarr version and retrieve the multiscales list
    if "ome" in attrs:
        ngff_version = attrs["ome"].get("version", "0.5")
        if not ngff_version.startswith("0.5"):
            raise ValueError(
                f"On-disk OME-Zarr has unsupported version '{ngff_version}'. "
                "Supported versions are 0.4, 0.5, 0.6 and 0.6.dev4."
            )
        multiscales = attrs["ome"]["multiscales"]
    elif "multiscales" in attrs:
        multiscales = attrs["multiscales"]
        ngff_version_in_meta = multiscales[0].get("version", "0.4")
        if not ngff_version_in_meta.startswith("0.4"):
            raise ValueError(
                f"On-disk OME-Zarr has unsupported multiscales version "
                f"'{ngff_version_in_meta}'. Supported versions are 0.4, 0.5, 0.6 and 0.6.dev4."
            )
        ngff_version = "0.4"
    else:
        raise ValueError(
            f"No OME-Zarr multiscales metadata found in {zarr_path}."
        )

    scale_keys = msi_utils.get_sorted_scale_keys(msim)
    n_levels_msim = len(scale_keys)
    n_levels_disk = len(multiscales[0]["datasets"])
    if n_levels_msim != n_levels_disk:
        raise ValueError(
            f"Number of resolution levels in msim ({n_levels_msim}) does not "
            f"match on-disk OME-Zarr ({n_levels_disk})."
        )

    sim0 = msi_utils.get_sim_from_msim(msim, scale=scale_keys[0])
    nsdims = si_utils.get_nonspatial_dims_from_sim(sim0)
    sdims = si_utils.get_spatial_dims_from_sim(sim0)

    for iscale, scale_key in enumerate(scale_keys):
        sim = msi_utils.get_sim_from_msim(msim, scale=scale_key)
        ngff_im = sim_to_ngff_image(sim, transform_key=transform_key)

        new_coordtfs = [
            {
                "type": "scale",
                "scale": [1.0] * len(nsdims)
                + [float(ngff_im.scale[dim]) for dim in sdims],
            },
            {
                "type": "translation",
                "translation": [0.0] * len(nsdims)
                + [float(ngff_im.translation[dim]) for dim in sdims],
            },
        ]
        multiscales[0]["datasets"][iscale]["coordinateTransformations"] = (
            new_coordtfs
        )

    # Write back only the "multiscales" key, leaving all other metadata intact
    if ngff_version.startswith("0.5"):
        # "multiscales" lives inside the "ome" namespace; read-modify-write
        # only that sub-key so that other "ome" entries (e.g. "omero") survive
        ome = dict(root.attrs["ome"])
        ome["multiscales"] = multiscales
        root.attrs["ome"] = ome
    else:
        # "multiscales" is a top-level attr in v0.4
        root.attrs["multiscales"] = multiscales


def read_msim_from_ome_zarr(
    zarr_path,
    transform_key=si_utils.DEFAULT_TRANSFORM_KEY,
    array_backend="zarr",
    target_coordinate_system=None,
):
    """Read a MultiscaleSpatialImage (xarray.DataTree).

    Supports OME-Zarr 0.4/0.5 and the 0.6.dev4 model, including its "0.6"
    version alias. In 0.4/0.5, scale and translation define sim coordinates;
    the separate affine under transform_key is identity. The 0.6 adapter can
    also import an additional coordinate transformation as a spatial affine.

    Parameters
    ----------
    zarr_path : str, Path or zarr store
        Image group to read. Parent scene groups are not traversed.
    transform_key : str, optional
        Package key under which to store the affine; defaults to
        si_utils.DEFAULT_TRANSFORM_KEY ("affine_metadata").
    array_backend : {"zarr", "dask"}, optional
        Lazy array backend, by default "zarr".
    target_coordinate_system : str or None, optional
        OME-Zarr coordinate-system name, independent of transform_key. None
        selects the sole supported additional transformation, or identity if
        none is present. Multiple candidates require explicit selection.
        Selecting the intrinsic coordinate system retains calibration only.

    Returns
    -------
    xarray.DataTree
        MultiscaleSpatialImage (msim) with the affine stored under transform_key.
    """
    if array_backend not in ("dask", "zarr"):
        raise ValueError("array_backend must be 'dask' or 'zarr'.")

    ngff_multiscales = read_ngff_multiscales(zarr_path)

    data_arrays = None
    if array_backend == "zarr":
        data_arrays = _open_ngff_dataset_arrays(zarr_path, ngff_multiscales)

    msim = ngff_multiscales_to_msim(
        ngff_multiscales,
        transform_key=transform_key,
        data_arrays=data_arrays,
        target_coordinate_system=target_coordinate_system,
    )

    # get channel names from omero metadata if available
    root = _zarr_compat.open_zarr_group(zarr_path, mode="r")
    omero = root.attrs.get("ome", {}).get("omero", root.attrs.get("omero"))
    if omero is not None:
        ch_coords = [ch["label"] for ch in omero["channels"]]
        if "c" in msim['scale0']["image"].dims:
            # A closure keeps this working across xarray releases:
            # DataTree.map_over_datasets() only grew its `kwargs` parameter
            # after the version shipped with Pyodide.
            def _assign_channel_coords(ds):
                return ds.assign_coords(c=ch_coords)

            msim = msim.map_over_datasets(_assign_channel_coords)

        # Display metadata is part of the image, not just a source-side aid
        # for recovering channel labels. Keeping it on the DataTree lets
        # virtual OME-Zarr views and derived outputs (notably browser fusion)
        # inherit the input colors and contrast windows.
        msim.attrs["omero"] = deepcopy(omero)

    return msim
