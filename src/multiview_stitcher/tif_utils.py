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

        with tifffile.TiffFile(self.path) as tif:
            self.n_pages = len(tif.pages)
            if self.n_pages == 0:
                raise ValueError("TIFF contains no pages")

            self.page_shape = tuple(tif.pages[0].shape)
            self.dtype = np.dtype(tif.pages[0].dtype)

            for page in tif.pages:
                if tuple(page.shape) != self.page_shape:
                    raise ValueError("All TIFF pages must have the same shape")
                if np.dtype(page.dtype) != self.dtype:
                    raise ValueError("All TIFF pages must have the same dtype")

        self.shape = (self.n_pages, *self.page_shape)
        self.chunks = (1, *self.page_shape)
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
        # threading.local isn't copyable/picklable; sims get deep-copied
        # (e.g. in msi_utils.get_msim_from_sim) and the store may cross
        # process boundaries with some dask schedulers, so drop it here and
        # let each copy/process lazily build its own cache.
        state = self.__dict__.copy()
        state.pop("_thread_local", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._thread_local = threading.local()

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

        suffix = "/".join(["0"] * (self.ndim - 1))
        for i in range(self.n_pages):
            yield f"c/{i}/{suffix}"

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
        #   c/<z>/0/0/...
        parts = key.split("/")

        if len(parts) != self.ndim + 1 or parts[0] != "c":
            return None

        try:
            chunk_indices = tuple(int(p) for p in parts[1:])
        except ValueError:
            return None

        z = chunk_indices[0]

        if not 0 <= z < self.n_pages:
            return None

        if any(i != 0 for i in chunk_indices[1:]):
            return None

        return z

    def _get_thread_local_tif(self):
        tif = getattr(self._thread_local, "tif", None)
        if tif is None:
            tif = tifffile.TiffFile(self.path)
            self._thread_local.tif = tif
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
    z = zarr.open_array(store=store, mode="r")
    return z, store


def tif_to_dask_plane_chunks(path):
    """
    Lazily read a TIFF as a Dask array, with one Dask chunk per TIFF plane/page.

    Returns shape:
        (n_planes, y, x)              for grayscale planes
        (n_planes, y, x, channels)    for RGB/multichannel planes
    """
    path = Path(path)

    # Read metadata only
    with tifffile.TiffFile(path) as tif:
        n_planes = len(tif.pages)
        plane_shape = tif.pages[0].shape
        dtype = tif.pages[0].dtype

        for page in tif.pages:
            if page.shape != plane_shape:
                raise ValueError("All TIFF planes must have the same shape.")
            if page.dtype != dtype:
                raise ValueError("All TIFF planes must have the same dtype.")

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

    return da.concatenate(planes, axis=0)
