from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import numpy as np
import tifffile
import zarr
from zarr.abc.store import Store

import dask.array as da
from dask import delayed


def _get_tiff_layout(tif):
    """
    Derive the page layout of an open TiffFile.

    Pages are written/read in row-major order over the non-spatial
    (e.g. t/c/z) axes, as declared by the series shape. This splits that
    leading part of the series shape off from the per-page (spatial) shape,
    so that e.g. a Z*C multi-page TIFF can be exposed with separate Z and C
    axes instead of a single flattened page axis.
    """
    n_pages = len(tif.pages)
    if n_pages == 0:
        raise ValueError("TIFF contains no pages")

    plane_shape = tuple(tif.pages[0].shape)
    dtype = np.dtype(tif.pages[0].dtype)

    for page in tif.pages:
        if tuple(page.shape) != plane_shape:
            raise ValueError("All TIFF pages must have the same shape")
        if np.dtype(page.dtype) != dtype:
            raise ValueError("All TIFF pages must have the same dtype")

    series_shape = tuple(tif.series[0].shape)
    non_spatial_shape = series_shape[: len(series_shape) - len(plane_shape)]

    expected_n_pages = int(np.prod(non_spatial_shape)) if non_spatial_shape else 1
    if expected_n_pages != n_pages:
        raise ValueError(
            "TIFF series shape is inconsistent with the number of pages"
        )

    return non_spatial_shape, plane_shape, dtype, n_pages


class TiffPagesZarrV3Store(Store):
    def __init__(self, path):
        super().__init__(read_only=True)
        self.path = Path(path)
        # Each thread gets its own cached, already-parsed TiffFile handle so
        # that repeated page reads (one per zarr chunk request) don't re-walk
        # the whole IFD chain every time. Re-parsing per read is cheap on
        # local disks but becomes the dominant cost (and can look like a
        # hang) for many-page TIFFs on network filesystems.
        self._thread_local = threading.local()
        # threading.local() only exposes the calling thread's own slot, so a
        # separate list tracks every handle opened across threads (zarr reads
        # pages via a shared thread pool) so close() can release them all.
        # Without this, cached handles keep the file open indefinitely, which
        # is harmless on POSIX but prevents deleting/moving the file on
        # Windows.
        self._open_handles = []
        self._open_handles_lock = threading.Lock()

        with tifffile.TiffFile(self.path) as tif:
            (
                self.non_spatial_shape,
                self.page_shape,
                self.dtype,
                self.n_pages,
            ) = _get_tiff_layout(tif)

        self.shape = (*self.non_spatial_shape, *self.page_shape)
        self.chunks = (1,) * len(self.non_spatial_shape) + self.page_shape
        self.ndim = len(self.shape)

        self.metadata = json.dumps(
            {
                "zarr_format": 3,
                "node_type": "array",
                "shape": list(self.shape),
                "data_type": self.dtype.name,
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {
                        "chunk_shape": list(self.chunks),
                    },
                },
                "chunk_key_encoding": {
                    "name": "default",
                    "configuration": {
                        "separator": "/",
                    },
                },
                "codecs": [
                    {
                        "name": "bytes",
                        "configuration": {
                            "endian": "little",
                        },
                    }
                ],
                "fill_value": 0,
                "attributes": {},
            }
        ).encode("utf-8")

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.path == other.path

    def __getstate__(self):
        # threading.local/Lock and open file handles aren't copyable/picklable;
        # sims get deep-copied (e.g. in msi_utils.get_msim_from_sim) and the
        # store may cross process boundaries with some dask schedulers, so
        # drop them here and let each copy/process lazily build its own cache.
        state = self.__dict__.copy()
        state.pop("_thread_local", None)
        state.pop("_open_handles", None)
        state.pop("_open_handles_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._thread_local = threading.local()
        self._open_handles = []
        self._open_handles_lock = threading.Lock()

    def close(self):
        super().close()
        lock = getattr(self, "_open_handles_lock", None)
        if lock is None:
            return
        with lock:
            handles, self._open_handles = self._open_handles, []
        for tif in handles:
            tif.close()

    def __del__(self):
        self.close()

    @property
    def supports_writes(self):
        return False

    @property
    def supports_deletes(self):
        return False

    @property
    def supports_listing(self):
        return True

    def with_read_only(self, read_only=True):
        return type(self)(self.path)

    async def get(self, key, prototype, byte_range=None):
        if key == "zarr.json":
            return prototype.buffer.from_bytes(self.metadata)

        page_index = self._page_index_from_key(key)
        if page_index is None:
            return None

        data = await asyncio.to_thread(self._read_page_bytes, page_index)
        return prototype.buffer.from_bytes(data)

    async def exists(self, key):
        return key == "zarr.json" or self._page_index_from_key(key) is not None

    async def list(self):
        yield "zarr.json"

        spatial_zeros = ["0"] * len(self.page_shape)
        for i in range(self.n_pages):
            leading = (
                [str(idx) for idx in np.unravel_index(i, self.non_spatial_shape)]
                if self.non_spatial_shape
                else []
            )
            yield "/".join(["c", *leading, *spatial_zeros])

    async def list_prefix(self, prefix):
        async for key in self.list():
            if key.startswith(prefix):
                yield key

    async def list_dir(self, prefix):
        prefix = prefix.strip("/")
        prefix = f"{prefix}/" if prefix else ""

        seen = set()
        async for key in self.list_prefix(prefix):
            rest = key[len(prefix):]
            if rest:
                seen.add(rest.split("/", 1)[0])

        for item in seen:
            yield item

    async def get_partial_values(self, prototype, key_ranges):
        return [
            await self.get(key, prototype=prototype, byte_range=byte_range)
            for key, byte_range in key_ranges
        ]

    async def set(self, key, value):
        raise NotImplementedError("read-only")

    async def delete(self, key):
        raise NotImplementedError("read-only")

    def _page_index_from_key(self, key):
        # Root-array Zarr v3 chunk keys look like:
        #   c/<i0>/<i1>/.../0/0/...
        # where the leading indices address the non-spatial (t/c/z/...) axes
        # and the trailing zeros address the (single) chunk per page.
        parts = key.split("/")

        if len(parts) != self.ndim + 1 or parts[0] != "c":
            return None

        try:
            chunk_indices = tuple(int(p) for p in parts[1:])
        except ValueError:
            return None

        n_leading = len(self.non_spatial_shape)
        leading_indices = chunk_indices[:n_leading]
        spatial_indices = chunk_indices[n_leading:]

        if any(i != 0 for i in spatial_indices):
            return None

        if not self.non_spatial_shape:
            return 0

        if any(
            not (0 <= idx < dim)
            for idx, dim in zip(leading_indices, self.non_spatial_shape)
        ):
            return None

        return int(np.ravel_multi_index(leading_indices, self.non_spatial_shape))

    def _get_thread_local_tif(self):
        tif = getattr(self._thread_local, "tif", None)
        if tif is None:
            tif = tifffile.TiffFile(self.path)
            self._thread_local.tif = tif
            with self._open_handles_lock:
                self._open_handles.append(tif)
        return tif

    def _read_page_bytes(self, i):
        tif = self._get_thread_local_tif()
        arr = tif.pages[i].asarray()

        arr = np.asarray(arr)

        # The metadata declares little-endian bytes.
        # For native little-endian arrays this is a no-op.
        if arr.dtype.itemsize > 1 and arr.dtype.byteorder not in ("<", "="):
            arr = arr.astype(arr.dtype.newbyteorder("<"), copy=False)

        return np.ascontiguousarray(arr).tobytes(order="C")


def tif_to_virtual_zarr_v3_plane_chunks(path):
    store = TiffPagesZarrV3Store(path)
    return zarr.open_array(store=store, mode="r")


def tif_to_dask_plane_chunks(path):
    """
    Lazily read a TIFF as a Dask array, with one Dask chunk per TIFF plane/page.

    The non-spatial (t/c/z/...) axes are inferred from the TIFF series shape,
    so e.g. a Z*C multi-page TIFF is returned with separate Z and C axes
    rather than a single flattened plane axis.
    """
    path = Path(path)

    # Read metadata only
    with tifffile.TiffFile(path) as tif:
        non_spatial_shape, plane_shape, dtype, n_planes = _get_tiff_layout(tif)

    @delayed
    def read_plane(i):
        with tifffile.TiffFile(path) as tif:
            return tif.pages[i].asarray()

    planes = [
        da.from_delayed(
            read_plane(i),
            shape=plane_shape,
            dtype=dtype,
        )[None, ...]
        for i in range(n_planes)
    ]

    data = da.concatenate(planes, axis=0)

    return data.reshape(*non_spatial_shape, *plane_shape)
