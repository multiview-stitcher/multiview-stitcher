"""
Dimension-name-agnostic virtual zarr array transformations.

This module owns *payload-only* numeric shape transforms of zarr arrays:
singleton axis expansion, stacking along a new axis, and chunk-aligned
concatenation along an existing axis. Each transform is expressed purely as a
remapping of output chunk keys onto source chunk keys and returns a **real**
``zarr.Array`` (opened on a small in-memory read-only store), not a generic
array-like shim.

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
* Serving chunks needs zarr-python v3's async store API, so the whole module is
  a no-op under zarr v2: the ``is_*`` predicates report False and callers take
  their eager fallback. See :func:`supports_virtual_arrays`.
"""

import numpy as np

from multiview_stitcher._zarr_compat import ZARR_V3, require_zarr_v3
from multiview_stitcher._zarr_compat import (
    codec_signature as _codec_signature,
)
from multiview_stitcher._zarr_compat import (
    open_virtual_array as _open_virtual,
)


class NotChunkAlignedError(ValueError):
    """Raised when a concat cannot be expressed as a pure chunk-key remap."""


def supports_virtual_arrays():
    """True when the transforms in this module are available.

    They rest on a read-only async store, which only zarr-python v3 has. Under
    v2 every caller has an eager equivalent (``xr.concat``, ``expand_dims``,
    ``da.from_zarr``), so callers should branch on this rather than let the
    transforms raise.
    """
    return ZARR_V3


# ---------------------------------------------------------------------------
# Public transforms
# ---------------------------------------------------------------------------


def expand_dims(zarray, n_leading_singletons):
    """Prepend ``n_leading_singletons`` size-1 axes to ``zarray`` (chunk 1 each).

    Returns a real ``zarr.Array`` of rank ``zarray.ndim + n`` whose leading axes
    are singleton. Each output chunk drops the (all-zero) leading chunk indices
    and reads the corresponding source chunk unchanged.
    """
    n = int(n_leading_singletons)
    if n <= 0:
        return zarray

    require_zarr_v3("Lazily expanding a zarr array")

    out_shape = (1,) * n + tuple(zarray.shape)
    out_chunks = (1,) * n + tuple(zarray.chunks)

    def dispatch(coords):
        return zarray, coords[n:]

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

    require_zarr_v3("Lazily stacking zarr arrays")

    first = zarrays[0]
    for other in zarrays[1:]:
        if tuple(other.shape) != tuple(first.shape):
            raise ValueError("stack requires identical shapes.")
        if tuple(other.chunks) != tuple(first.chunks):
            raise ValueError("stack requires identical chunks.")
        if _codec_signature(other) != _codec_signature(first):
            raise ValueError("stack requires identical dtype/codecs.")

    axis = int(axis)
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
    if not supports_virtual_arrays():
        return False
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
    axis = int(axis)
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
    if not supports_virtual_arrays():
        return False
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
    require_zarr_v3("Lazily concatenating zarr arrays")

    zarrays = list(zarrays)
    axis = int(axis)
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
