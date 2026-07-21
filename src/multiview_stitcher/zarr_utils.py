"""
Dimension-name-agnostic virtual zarr array transformations.

This module owns *payload-only* numeric shape transforms of zarr arrays:
singleton axis expansion, scalar indexing on chunk-size-1 axes, transpose of
chunk-size-1 axes, stacking along a new axis, and chunk-aligned concatenation
along an existing axis. Each transform is expressed purely as a remapping of
output chunk keys onto source chunk keys and returns a **real** ``zarr.Array``
(opened on a small in-memory read-only store), not a generic array-like shim.

Design contract
---------------
* Everything here knows only about *numbers*: shape, chunks, dtype, codecs and
  chunk-key dispatch. It has no notion of dimension names, coordinates, ``t``/
  ``c`` axes or xarray. Labelled-dimension semantics live in
  ``spatial_image_utils``/``msi_utils``.
* Chunk contents are served by **byte passthrough**: an output chunk maps to
  exactly one source chunk, and the source's already-encoded bytes are returned
  unchanged. There is no decode/re-encode and no materialization. This is only
  valid for chunk-key-remappable transforms, which is why the output array
  mirrors the source's ``zarr_format``, dtype and codecs and only overrides the
  shape and chunk grid.
* Because these transforms only ever combine whole chunks, they compose: a
  virtual array may itself be a source of another virtual array.
"""

import json

import numpy as np
import zarr
from zarr.abc.store import Store
from zarr.core.buffer import default_buffer_prototype


class NotChunkAlignedError(ValueError):
    """Raised when a transform cannot be expressed as a chunk-key remap."""


def _json_default(obj):
    # zarr metadata dicts may contain numpy scalars / arrays and tuples; make
    # them JSON serialisable while preserving their values.
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def _encode_source_chunk_key(zarray, coords):
    """Return the store key of a source chunk in the source's own encoding.

    zarr v3 exposes chunk-key encoding via ``metadata.chunk_key_encoding`` while
    zarr v2 exposes ``metadata.encode_chunk_key`` directly; support both so that
    v2 (OME-Zarr 0.4) and v3 sources work transparently.
    """
    encoding = getattr(zarray.metadata, "chunk_key_encoding", None)
    if encoding is not None:
        key = encoding.encode_chunk_key(coords)
    else:
        key = zarray.metadata.encode_chunk_key(coords)
    # Prefix with the array's own path within its store (usually empty when the
    # array was opened at the store root, as is the case for our sources).
    return f"{zarray.path}/{key}" if zarray.path else key


def _codec_signature(zarray):
    """Return a hashable signature of everything that must match for passthrough.

    Two arrays can be stacked/concatenated by byte passthrough only if their
    encoded chunk bytes are mutually decodable by a single output array, i.e.
    they share format, dtype, fill value and codec pipeline. Chunk *shape* and
    array *shape* are checked separately by the callers.
    """
    md = zarray.metadata.to_dict()

    def _norm(value):
        # Normalise nested codec/config dicts to a canonical JSON string.
        return json.dumps(value, sort_keys=True, default=_json_default)

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


class _VirtualZarrStore(Store):
    """Read-only in-memory store serving a single synthesized zarr array.

    It answers exactly two kinds of requests:

    * the array metadata key (``zarr.json`` for v3, ``.zarray`` for v2), returned
      from the synthesized metadata bytes, and
    * chunk keys, resolved via ``dispatch`` to a ``(source_array, source_coords)``
      pair whose already-encoded bytes are streamed straight through.

    The store is intentionally dimension-agnostic; it receives an opaque
    ``dispatch`` callable and a chunk-grid shape (only used for listing).
    """

    supports_writes = False
    supports_deletes = False
    supports_listing = True
    supports_partial_writes = False

    def __init__(self, meta_bytes, meta_key, dispatch, grid_shape):
        super().__init__(read_only=True)
        self._meta_bytes = meta_bytes
        self._meta_key = meta_key
        # dispatch: tuple[int, ...] -> (zarr.Array, tuple[int, ...]) | None
        self._dispatch = dispatch
        self._grid_shape = tuple(int(g) for g in grid_shape)

    # Identity equality: each virtual array owns its own store instance.
    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)

    async def get(self, key, prototype, byte_range=None):
        if key == self._meta_key:
            return prototype.buffer.from_bytes(self._meta_bytes)
        # v2 may probe for optional array attributes; report "no attrs".
        if key == ".zattrs":
            return prototype.buffer.from_bytes(b"{}")

        coords = self._parse_chunk_key(key)
        if coords is None:
            return None

        target = self._dispatch(coords)
        if target is None:
            return None

        source_array, source_coords = target
        source_key = _encode_source_chunk_key(source_array, source_coords)
        # Byte passthrough: the source store already holds encoded chunk bytes.
        return await source_array.store.get(source_key, prototype, byte_range)

    async def get_partial_values(self, prototype, key_ranges):
        return [
            await self.get(key, prototype, byte_range)
            for key, byte_range in key_ranges
        ]

    async def exists(self, key):
        return (await self.get(key, default_buffer_prototype())) is not None

    async def set(self, key, value):  # noqa: A003  # pragma: no cover
        raise NotImplementedError("virtual zarr store is read-only")

    async def delete(self, key):  # pragma: no cover - read-only store
        raise NotImplementedError("virtual zarr store is read-only")

    async def list(self):  # noqa: A003 - zarr Store interface
        yield self._meta_key
        if not self._grid_shape:
            yield "c" if self._meta_key == "zarr.json" else "0"
            return

        prefix = "c/" if self._meta_key == "zarr.json" else ""
        for coords in np.ndindex(*self._grid_shape):
            yield prefix + "/".join(str(c) for c in coords)

    async def list_dir(self, prefix):
        seen = set()
        async for key in self.list():
            if prefix and not key.startswith(prefix):
                continue
            top = key[len(prefix):].lstrip("/").split("/")[0]
            if top and top not in seen:
                seen.add(top)
                yield top

    async def list_prefix(self, prefix):
        async for key in self.list():
            if key.startswith(prefix):
                yield key

    def _parse_chunk_key(self, key):
        # Output arrays always use separator "/"; v3 additionally prefixes "c/".
        if not self._grid_shape and key in {"0", "c"}:
            return ()
        body = key[2:] if key.startswith("c/") else key
        try:
            return tuple(int(part) for part in body.split("/"))
        except ValueError:
            return None


def _synthesize_metadata(template, out_shape, out_chunks):
    """Build metadata bytes + metadata key for a virtual array.

    The output mirrors ``template``'s format, dtype and codecs and overrides only
    the shape and chunk grid, and forces a "/"-separated (default) chunk-key
    encoding so the store receives predictable keys.
    """
    md = dict(template.metadata.to_dict())
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

    meta_bytes = json.dumps(md, default=_json_default).encode("utf-8")
    return meta_bytes, meta_key


def _grid_shape(shape, chunks):
    return tuple(
        int(np.ceil(s / c)) if c else 0 for s, c in zip(shape, chunks)
    )


def _open_virtual(template, out_shape, out_chunks, dispatch):
    """Open a real, read-only ``zarr.Array`` over a virtual store."""
    meta_bytes, meta_key = _synthesize_metadata(template, out_shape, out_chunks)
    store = _VirtualZarrStore(
        meta_bytes, meta_key, dispatch, _grid_shape(out_shape, out_chunks)
    )
    return zarr.open_array(store=store, mode="r")


# ---------------------------------------------------------------------------
# Public transforms
# ---------------------------------------------------------------------------


def _normalize_axis(axis, ndim):
    axis = int(axis)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise IndexError(
            f"axis {axis} is out of bounds for array of dimension {ndim}"
        )
    return axis


def _normalize_axes(axes, ndim):
    axes = (axes,) if np.isscalar(axes) else tuple(axes)
    normalized = tuple(_normalize_axis(axis, ndim) for axis in axes)
    if len(set(normalized)) != len(normalized):
        raise ValueError("repeated axis")
    return normalized


def expand_dims(zarray, n_leading_singletons):
    """Prepend ``n_leading_singletons`` size-1 axes to ``zarray`` (chunk 1 each).

    Returns a real ``zarr.Array`` of rank ``zarray.ndim + n`` whose leading axes
    are singleton. Each output chunk drops the (all-zero) leading chunk indices
    and reads the corresponding source chunk unchanged.
    """
    n = int(n_leading_singletons)
    if n <= 0:
        return zarray

    out_shape = (1,) * n + tuple(zarray.shape)
    out_chunks = (1,) * n + tuple(zarray.chunks)

    def dispatch(coords):
        return zarray, coords[n:]

    return _open_virtual(zarray, out_shape, out_chunks, dispatch)


def expand_dims_at(zarray, axes):
    """Insert singleton axes at explicit positions without reading data.

    ``axes`` follows :func:`numpy.expand_dims`: positions refer to the output
    rank and may be negative. This complements :func:`expand_dims`, whose
    integer argument is retained as the number of leading axes for backward
    compatibility.
    """
    axes = tuple(axes) if not np.isscalar(axes) else (axes,)
    if not axes:
        return zarray

    out_ndim = zarray.ndim + len(axes)
    axes = _normalize_axes(axes, out_ndim)
    axis_set = set(axes)

    source_shape = iter(zarray.shape)
    source_chunks = iter(zarray.chunks)
    out_shape = tuple(
        1 if axis in axis_set else next(source_shape)
        for axis in range(out_ndim)
    )
    out_chunks = tuple(
        1 if axis in axis_set else next(source_chunks)
        for axis in range(out_ndim)
    )

    def dispatch(coords):
        return zarray, tuple(
            coord for axis, coord in enumerate(coords) if axis not in axis_set
        )

    return _open_virtual(zarray, out_shape, out_chunks, dispatch)


def getitem(zarray, selection):
    """Apply a basic scalar selection by remapping whole chunks.

    Integer indices remove their axes, matching NumPy ``__getitem__``. Since an
    encoded output chunk must map to exactly one encoded source chunk, integer
    indexing is supported only on axes with chunk size 1. All retained axes
    must use a full slice. These constraints cover lazy channel/time selection
    while keeping the transform decode-free and composable.
    """
    if not isinstance(selection, tuple):
        selection = (selection,)

    ellipsis_count = sum(item is Ellipsis for item in selection)
    if ellipsis_count > 1:
        raise IndexError("an index can only have a single ellipsis")
    if ellipsis_count:
        ellipsis_index = selection.index(Ellipsis)
        missing = zarray.ndim - (len(selection) - 1)
        if missing < 0:
            raise IndexError("too many indices for array")
        selection = (
            selection[:ellipsis_index]
            + (slice(None),) * missing
            + selection[ellipsis_index + 1 :]
        )
    elif len(selection) < zarray.ndim:
        selection = selection + (slice(None),) * (zarray.ndim - len(selection))

    if len(selection) != zarray.ndim:
        raise IndexError("incorrect number of indices for array")

    normalized = []
    retained_axes = []
    for axis, (item, size, chunk) in enumerate(
        zip(selection, zarray.shape, zarray.chunks)
    ):
        if isinstance(item, (int, np.integer)) and not isinstance(
            item, (bool, np.bool_)
        ):
            index = int(item)
            if index < 0:
                index += int(size)
            if index < 0 or index >= size:
                raise IndexError(
                    f"index {item} is out of bounds for axis {axis} "
                    f"with size {size}"
                )
            if chunk != 1:
                raise NotChunkAlignedError(
                    f"integer indexing requires chunk size 1 on axis {axis}; "
                    f"got {chunk}."
                )
            normalized.append(index)
            continue

        if not (
            isinstance(item, slice)
            and item.indices(int(size)) == (0, int(size), 1)
        ):
            raise NotChunkAlignedError(
                "retained axes must be selected with a full slice."
            )
        normalized.append(slice(None))
        retained_axes.append(axis)

    out_shape = tuple(zarray.shape[axis] for axis in retained_axes)
    out_chunks = tuple(zarray.chunks[axis] for axis in retained_axes)

    def dispatch(coords):
        source_coords = []
        output_coords = iter(coords)
        for item in normalized:
            # A scalar index is also its source chunk coordinate because scalar
            # indexing is restricted to axes whose chunks have size one.
            source_coords.append(
                item if isinstance(item, int) else next(output_coords)
            )
        return zarray, tuple(source_coords)

    return _open_virtual(zarray, out_shape, out_chunks, dispatch)


def is_transposable(zarray, axes=None):
    """Return whether ``transpose`` can pass encoded chunks through unchanged."""
    if axes is None:
        axes = tuple(reversed(range(zarray.ndim)))
    try:
        axes = _normalize_axes(axes, zarray.ndim)
    except (IndexError, ValueError):
        return False
    if len(axes) != zarray.ndim:
        return False

    # Moving axes whose chunks are singleton does not change the flattened
    # contents of a chunk. Axes with larger chunks must retain relative order.
    fixed_axes = [
        axis for axis, chunk in enumerate(zarray.chunks) if chunk != 1
    ]
    return [axis for axis in axes if zarray.chunks[axis] != 1] == fixed_axes


def transpose(zarray, axes=None):
    """Permute axes when only chunk-size-1 axes change relative position.

    The restriction guarantees that the C-order contents of every encoded chunk
    remain unchanged, allowing a pure chunk-key remap. It covers canonicalizing
    channel/time axes while rejecting spatial transposes that require decoding.
    """
    if axes is None:
        axes = tuple(reversed(range(zarray.ndim)))
    axes = _normalize_axes(axes, zarray.ndim)
    if len(axes) != zarray.ndim:
        raise ValueError("axes must contain one entry per source dimension")
    if not is_transposable(zarray, axes):
        raise NotChunkAlignedError(
            "transpose may only move axes with chunk size 1."
        )
    if axes == tuple(range(zarray.ndim)):
        return zarray

    out_shape = tuple(zarray.shape[axis] for axis in axes)
    out_chunks = tuple(zarray.chunks[axis] for axis in axes)

    def dispatch(coords):
        # Invert the output-to-source axis permutation to recover the source
        # chunk key without touching the encoded chunk contents.
        source_coords = [None] * zarray.ndim
        for output_axis, source_axis in enumerate(axes):
            source_coords[source_axis] = coords[output_axis]
        return zarray, tuple(source_coords)

    return _open_virtual(zarray, out_shape, out_chunks, dispatch)


def stack(zarrays, axis=0):
    """Stack ``zarrays`` along a new ``axis`` with chunk size 1.

    All inputs must share shape, chunks, dtype and codecs. Returns a real
    ``zarr.Array`` whose new axis has size ``len(zarrays)`` and chunk size 1;
    output chunk ``(..., i, ...)`` (with ``i`` at position ``axis``) maps to
    source ``i`` at the remaining chunk coordinates. ``axis`` is configurable so
    callers can place the new dimension in a canonical order without a lazy
    transpose that would break the one-real-array-per-DataArray invariant.
    """
    zarrays = list(zarrays)
    if not zarrays:
        raise ValueError("stack requires at least one array.")

    first = zarrays[0]
    for other in zarrays[1:]:
        if tuple(other.shape) != tuple(first.shape):
            raise ValueError("stack requires identical shapes.")
        if tuple(other.chunks) != tuple(first.chunks):
            raise ValueError("stack requires identical chunks.")
        if _codec_signature(other) != _codec_signature(first):
            raise ValueError("stack requires identical dtype/codecs.")

    axis = _normalize_axis(axis, first.ndim + 1)
    shape = tuple(first.shape)
    chunks = tuple(first.chunks)
    out_shape = shape[:axis] + (len(zarrays),) + shape[axis:]
    out_chunks = chunks[:axis] + (1,) + chunks[axis:]

    def dispatch(coords):
        index = coords[axis]
        if index < 0 or index >= len(zarrays):
            return None
        rest = coords[:axis] + coords[axis + 1:]
        return zarrays[index], rest

    return _open_virtual(first, out_shape, out_chunks, dispatch)


def is_stackable(zarrays):
    """Return True when ``stack(zarrays)`` would succeed.

    Lets callers fall back to an eager path (mirroring
    :func:`is_chunk_aligned_concatenate`) instead of hitting a ``ValueError``.
    """
    zarrays = list(zarrays)
    if not zarrays:
        return False
    first = zarrays[0]
    for other in zarrays[1:]:
        if tuple(other.shape) != tuple(first.shape):
            return False
        if tuple(other.chunks) != tuple(first.chunks):
            return False
        if _codec_signature(other) != _codec_signature(first):
            return False
    return True


def _concatenate_layout(zarrays, axis):
    """Validate a concat and return ``(out_shape, out_chunks, cum_counts)``.

    Raises :class:`NotChunkAlignedError` when the concat cannot be expressed as a
    pure chunk-key remap (incompatible arrays, or a source other than the last
    whose extent along ``axis`` is not a whole number of chunks).
    """
    zarrays = list(zarrays)
    if not zarrays:
        raise ValueError("concatenate requires at least one array.")

    first = zarrays[0]
    axis = _normalize_axis(axis, first.ndim)
    chunk = int(first.chunks[axis])

    for other in zarrays[1:]:
        if tuple(other.chunks) != tuple(first.chunks):
            raise NotChunkAlignedError("concat requires identical chunks.")
        if _codec_signature(other) != _codec_signature(first):
            raise NotChunkAlignedError("concat requires identical dtype/codecs.")
        # Every axis except the concat axis must match in size.
        for ax, (s0, s1) in enumerate(zip(first.shape, other.shape)):
            if ax != axis and s0 != s1:
                raise NotChunkAlignedError(
                    "concat requires equal shapes off the concat axis."
                )

    # Chunk alignment: every source but the last must end on a chunk boundary,
    # otherwise output chunks would straddle two sources.
    counts = []
    for i, z in enumerate(zarrays):
        size = int(z.shape[axis])
        if i != len(zarrays) - 1 and size % chunk != 0:
            raise NotChunkAlignedError(
                f"source {i} extent {size} along axis {axis} is not a multiple "
                f"of chunk size {chunk}."
            )
        counts.append(int(np.ceil(size / chunk)))

    cum_counts = np.cumsum([0] + counts)
    out_shape = list(first.shape)
    out_shape[axis] = sum(int(z.shape[axis]) for z in zarrays)
    return tuple(out_shape), tuple(first.chunks), cum_counts


def is_chunk_aligned_concatenate(zarrays, axis):
    """Return True when ``concatenate(zarrays, axis)`` would succeed."""
    try:
        _concatenate_layout(zarrays, axis)
    except NotChunkAlignedError:
        return False
    return True


def concatenate(zarrays, axis):
    """Concatenate ``zarrays`` along an existing ``axis`` by chunk-key remap.

    Only valid when chunk-aligned (see :func:`is_chunk_aligned_concatenate`): axes
    such as ``c``/``t`` with chunk size 1 always qualify. Output chunk index
    ``k`` along ``axis`` is routed to the owning source and its local chunk
    index; all other chunk indices pass through unchanged.
    """
    zarrays = list(zarrays)
    axis = _normalize_axis(axis, zarrays[0].ndim)
    out_shape, out_chunks, cum_counts = _concatenate_layout(zarrays, axis)

    def dispatch(coords):
        k = coords[axis]
        # Locate the source whose chunk range contains output chunk k.
        source_index = int(np.searchsorted(cum_counts, k, side="right") - 1)
        if source_index < 0 or source_index >= len(zarrays):
            return None
        local = list(coords)
        local[axis] = k - int(cum_counts[source_index])
        return zarrays[source_index], tuple(local)

    return _open_virtual(zarrays[0], out_shape, out_chunks, dispatch)
