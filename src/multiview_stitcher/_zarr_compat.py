"""
Byte-level zarr helpers shared by both of multiview-stitcher's environments.

CPython and Pyodide now run the same zarr-python v3, so nothing here branches
on the library version. What it does branch on is the *format* of the data: an
OME-Zarr 0.4 store is a zarr v2 hierarchy and a 0.5 one is v3, and both are
read through the same v3 library.

Everything in this module is *numeric and byte-level*: array metadata dicts,
codec signatures, chunk-key encoding and read-only "virtual" arrays whose
chunks are served by remapping output chunk coordinates onto source chunks.
No dimension names, coordinates or xarray concepts appear here; those live in
``spatial_image_utils``/``msi_utils``.

The public helpers are used by :mod:`multiview_stitcher.zarr_utils`.
"""

import json
import os
from collections.abc import MutableMapping

import numcodecs
import numcodecs.registry
import numpy as np
import zarr


# ---------------------------------------------------------------------------
# Opening zarr objects from a path or a store
# ---------------------------------------------------------------------------
#
# A source may be a path/URL or an already-constructed store. The latter is
# what the browser runtime uses, to route reads through its service worker, and
# zarr spells the two cases differently.


def is_pathlike(source):
    return isinstance(source, (str, os.PathLike))


def open_zarr_group(source, mode="r", **kwargs):
    """Open a zarr group from a path/URL or from a store-like object."""
    if is_pathlike(source):
        return zarr.open_group(str(source), mode=mode, **kwargs)
    return zarr.open_group(store=source, mode=mode, **kwargs)


def open_zarr_array(source, path, mode="r", **kwargs):
    """Open a zarr array below ``source`` at the relative ``path``."""
    if is_pathlike(source):
        return zarr.open_array(
            os.path.join(str(source), str(path)), mode=mode, **kwargs
        )
    return zarr.open_array(store=source, path=str(path), mode=mode, **kwargs)


# ---------------------------------------------------------------------------
# Codec metadata written by other tools
# ---------------------------------------------------------------------------

#: Blosc settings that only describe how data was *compressed*. Blosc records
#: them in each compressed block's own header, so decompression re-reads them
#: from the bytes and never needs the value in the metadata. numcodecs grew a
#: parameter for `typesize` well after writers started emitting it - notably
#: bioformats2raw, whose OME-Zarrs carry it in every `.zarray` - so an older
#: build refuses a file it is perfectly able to read.
_BLOSC_ENCODE_ONLY_KEYS = frozenset({"typesize"})


def codec_from_config(base, config, droppable):
    """Build a codec from ``config``, dropping keys ``base`` cannot take.

    Only the keys in ``droppable`` may be dropped, and only once the codec has
    actually refused them: a key this library does not know about is left to
    fail, because silently ignoring one would decode the bytes wrongly rather
    than not at all.
    """
    try:
        return base.from_config(config)
    except TypeError:
        trimmed = {
            key: value
            for key, value in config.items()
            if key not in droppable
        }
        if trimmed == config:
            raise
        return base.from_config(trimmed)


class _CompatibleBlosc(numcodecs.Blosc):
    """Blosc that reads configurations written for a newer numcodecs."""

    @classmethod
    def from_config(cls, config):
        return codec_from_config(
            numcodecs.Blosc, config, _BLOSC_ENCODE_ONLY_KEYS
        )


def register_compatible_codecs():
    """Teach numcodecs to read codec metadata from other implementations.

    Registered process-wide, the way a codec plugin registers itself, because
    the decode happens deep inside zarr where no argument of ours reaches. The
    replacement differs from the original only in what it accepts, never in
    what it produces, and compression is untouched.
    """
    numcodecs.registry.register_codec(_CompatibleBlosc, "blosc")


register_compatible_codecs()


def json_default(obj):
    """JSON encoder fallback for zarr metadata dicts.

    zarr metadata may contain numpy scalars / arrays and tuples; make them
    JSON serialisable while preserving their values.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def array_metadata_dict(zarray):
    """Return ``zarray``'s own array metadata as a plain dict.

    Note that this is the metadata of the *data*, not of the library: an
    OME-Zarr 0.4 array read by zarr-python v3 reports ``zarr_format`` 2, and
    the callers below branch on that rather than on the library version.
    """
    return dict(zarray.metadata.to_dict())


def codec_signature(zarray):
    """Return a hashable signature of everything that must match for
    byte-passthrough between arrays.

    Two arrays can be stacked/concatenated by byte passthrough only if their
    encoded chunk bytes are mutually decodable by a single output array, i.e.
    they share format, dtype, fill value and codec pipeline. Chunk *shape* and
    array *shape* are checked separately by the callers.
    """
    md = array_metadata_dict(zarray)

    def _norm(value):
        # Normalise nested codec/config dicts to a canonical JSON string.
        return json.dumps(value, sort_keys=True, default=json_default)

    if md["zarr_format"] == 2:
        return (
            2,
            str(md.get("dtype")),
            _norm(md.get("compressor")),
            _norm(md.get("filters")),
            md.get("order"),
            _norm(md.get("fill_value")),
        )
    return (
        3,
        str(md.get("data_type")),
        _norm(md.get("codecs")),
        _norm(md.get("fill_value")),
    )


def encode_source_chunk_key(zarray, coords):
    """Return the store key of a source chunk in the source's own encoding.

    The returned key already includes the array's own path within its store
    (usually empty for arrays opened at the store root, as is the case for our
    sources).
    """
    encoding = getattr(zarray.metadata, "chunk_key_encoding", None)
    if encoding is not None:
        key = encoding.encode_chunk_key(coords)
    else:
        key = zarray.metadata.encode_chunk_key(coords)
    return f"{zarray.path}/{key}" if zarray.path else key


def synthesize_metadata(template, out_shape, out_chunks):
    """Build metadata bytes + metadata key for a virtual array.

    The output mirrors ``template``'s format, dtype and codecs and overrides
    only the shape and chunk grid, and forces a "/"-separated (default)
    chunk-key encoding so the store receives predictable keys.
    """
    md = array_metadata_dict(template)
    out_shape = [int(s) for s in out_shape]
    out_chunks = [int(c) for c in out_chunks]

    if md["zarr_format"] == 2:
        md["shape"] = out_shape
        md["chunks"] = out_chunks
        md["dimension_separator"] = "/"
        meta_key = ".zarray"
    else:
        md["shape"] = out_shape
        md["chunk_grid"] = {
            "name": "regular",
            "configuration": {"chunk_shape": out_chunks},
        }
        md["chunk_key_encoding"] = {
            "name": "default",
            "configuration": {"separator": "/"},
        }
        meta_key = "zarr.json"

    meta_bytes = json.dumps(md, default=json_default).encode("utf-8")
    return meta_bytes, meta_key


def grid_shape(shape, chunks):
    return tuple(
        int(np.ceil(s / c)) if c else 0 for s, c in zip(shape, chunks)
    )


def parse_virtual_chunk_key(key):
    """Parse a chunk key of a virtual array into chunk coordinates.

    Virtual arrays always use separator "/"; zarr v3 additionally prefixes
    chunk keys with "c/".
    """
    body = key[2:] if key.startswith("c/") else key
    try:
        return tuple(int(part) for part in body.split("/"))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Virtual stores
# ---------------------------------------------------------------------------


def _make_v3_store_class():
    """Build the zarr v3 virtual store class.

    Defined lazily inside a function so that importing this module under
    zarr v2 never touches the v3-only ``zarr.abc`` / ``zarr.core`` namespaces.
    """
    from zarr.abc.store import Store
    from zarr.core.buffer import default_buffer_prototype

    class _VirtualZarrStoreV3(Store):
        """Read-only in-memory store serving a single synthesized zarr array."""

        supports_writes = False
        supports_deletes = False
        supports_listing = True
        supports_partial_writes = False

        def __init__(self, meta_bytes, meta_key, dispatch, grid):
            super().__init__(read_only=True)
            self._meta_bytes = meta_bytes
            self._meta_key = meta_key
            # dispatch: tuple[int, ...] -> (zarr.Array, tuple[int, ...]) | None
            self._dispatch = dispatch
            self._grid_shape = tuple(int(g) for g in grid)

        # Identity equality: each virtual array owns its own store instance.
        def __eq__(self, other):
            return self is other

        def __hash__(self):
            return id(self)

        async def get(self, key, prototype, byte_range=None):
            if key == self._meta_key:
                return prototype.buffer.from_bytes(self._meta_bytes)
            # v2-format arrays may probe for optional array attributes.
            if key == ".zattrs":
                return prototype.buffer.from_bytes(b"{}")

            coords = parse_virtual_chunk_key(key)
            if coords is None:
                return None

            target = self._dispatch(coords)
            if target is None:
                return None

            source_array, source_coords = target
            source_key = encode_source_chunk_key(source_array, source_coords)
            # Byte passthrough: the source store already holds encoded bytes.
            return await source_array.store.get(
                source_key, prototype, byte_range
            )

        async def get_partial_values(self, prototype, key_ranges):
            return [
                await self.get(key, prototype, byte_range)
                for key, byte_range in key_ranges
            ]

        async def exists(self, key):
            return (await self.get(key, default_buffer_prototype())) is not None

        async def set(self, key, value):  # pragma: no cover - read-only store
            raise NotImplementedError("virtual zarr store is read-only")

        async def delete(self, key):  # pragma: no cover - read-only store
            raise NotImplementedError("virtual zarr store is read-only")

        async def list(self):
            yield self._meta_key
            for coords in np.ndindex(*self._grid_shape):
                yield "c/" + "/".join(str(c) for c in coords)

        async def list_dir(self, prefix):
            seen = set()
            async for key in self.list():
                if prefix and not key.startswith(prefix):
                    continue
                top = key[len(prefix) :].lstrip("/").split("/")[0]
                if top and top not in seen:
                    seen.add(top)
                    yield top

        async def list_prefix(self, prefix):
            async for key in self.list():
                if key.startswith(prefix):
                    yield key

    return _VirtualZarrStoreV3


_v3_store_class = None


def open_virtual_array(template, out_shape, out_chunks, dispatch):
    """Open a real, read-only ``zarr.Array`` over a virtual store.

    ``dispatch`` maps output chunk coordinates to a
    ``(source_zarr_array, source_chunk_coords)`` pair, or ``None`` for chunks
    that have no source (which read back as the fill value).
    """
    global _v3_store_class

    meta_bytes, meta_key = synthesize_metadata(template, out_shape, out_chunks)
    grid = grid_shape(out_shape, out_chunks)

    if _v3_store_class is None:
        _v3_store_class = _make_v3_store_class()

    store = _v3_store_class(meta_bytes, meta_key, dispatch, grid)
    return zarr.open_array(store=store, mode="r")
