import os
import tempfile

import numpy as np
import pytest
import zarr

from multiview_stitcher import zarr_utils


def _write_zarr(tmpdir, name, data, chunks, zarr_format):
    # OME-Zarr 0.4 arrays are zarr v2 (dimension_separator "/"); v0.5 are v3.
    kwargs = {"dimension_separator": "/"} if zarr_format == 2 else {}
    zarray = zarr.open_array(
        os.path.join(tmpdir, name),
        mode="w",
        shape=data.shape,
        chunks=chunks,
        dtype=data.dtype,
        zarr_format=zarr_format,
        **kwargs,
    )
    zarray[:] = data
    return zarray


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_virtual_expand_dims_returns_real_zarr_array(zarr_format):
    with tempfile.TemporaryDirectory() as tmpdir:
        data = np.arange(16, dtype=np.uint16).reshape(4, 4)
        source = _write_zarr(tmpdir, "a.zarr", data, (2, 2), zarr_format)

        expanded = zarr_utils.virtual_expand_dims(source, 2)

        # A real zarr.Array with two leading singleton axes (chunk 1 each).
        assert isinstance(expanded, zarr.Array)
        assert expanded.shape == (1, 1, 4, 4)
        assert expanded.chunks == (1, 1, 2, 2)
        assert expanded.dtype == data.dtype
        assert np.array_equal(expanded[0, 0], data)
        # Sub-slices read byte-exact from the source chunks.
        assert np.array_equal(expanded[0, 0, 1:3, 1:3], data[1:3, 1:3])

    # A zero expansion is a no-op passthrough.
    assert zarr_utils.virtual_expand_dims(source, 0) is source


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_virtual_stack_returns_real_zarr_array(zarr_format):
    with tempfile.TemporaryDirectory() as tmpdir:
        a = _write_zarr(
            tmpdir, "a.zarr", np.full((4, 4), 1, np.uint16), (2, 2), zarr_format
        )
        b = _write_zarr(
            tmpdir, "b.zarr", np.full((4, 4), 2, np.uint16), (2, 2), zarr_format
        )

        stacked = zarr_utils.virtual_stack([a, b])
        assert isinstance(stacked, zarr.Array)
        assert stacked.shape == (2, 4, 4)
        assert stacked.chunks == (1, 2, 2)
        assert np.array_equal(stacked[0], a[:])
        assert np.array_equal(stacked[1], b[:])

        # The new axis can be inserted at an arbitrary position.
        stacked_axis1 = zarr_utils.virtual_stack([a, b], axis=1)
        assert stacked_axis1.shape == (4, 2, 4)
        assert np.array_equal(stacked_axis1[:, 0], a[:])
        assert np.array_equal(stacked_axis1[:, 1], b[:])


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_virtual_concat_chunk_aligned(zarr_format):
    with tempfile.TemporaryDirectory() as tmpdir:
        a = _write_zarr(
            tmpdir, "a.zarr", np.full((4, 4), 1, np.uint16), (2, 2), zarr_format
        )
        b = _write_zarr(
            tmpdir, "b.zarr", np.full((4, 4), 2, np.uint16), (2, 2), zarr_format
        )

        assert zarr_utils.is_chunk_aligned_concat([a, b], axis=0)
        concatenated = zarr_utils.virtual_concat([a, b], axis=0)
        assert isinstance(concatenated, zarr.Array)
        assert concatenated.shape == (8, 4)
        assert np.array_equal(concatenated[:4], a[:])
        assert np.array_equal(concatenated[4:], b[:])


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_virtual_concat_chunk_size_one_axis_always_aligned(zarr_format):
    # Chunk-size-1 axes (e.g. c/t) are always chunk-aligned, whatever the sizes.
    with tempfile.TemporaryDirectory() as tmpdir:
        a = _write_zarr(
            tmpdir, "a.zarr", np.full((3, 4), 1, np.uint16), (1, 2), zarr_format
        )
        b = _write_zarr(
            tmpdir, "b.zarr", np.full((2, 4), 2, np.uint16), (1, 2), zarr_format
        )
        assert zarr_utils.is_chunk_aligned_concat([a, b], axis=0)
        concatenated = zarr_utils.virtual_concat([a, b], axis=0)
        assert concatenated.shape == (5, 4)
        assert np.array_equal(concatenated[:3], a[:])
        assert np.array_equal(concatenated[3:], b[:])


def test_virtual_concat_rejects_non_chunk_aligned():
    with tempfile.TemporaryDirectory() as tmpdir:
        # A non-final source whose extent is not a whole number of chunks.
        a = _write_zarr(tmpdir, "a.zarr", np.ones((3, 4), np.uint16), (2, 2), 3)
        b = _write_zarr(tmpdir, "b.zarr", np.ones((4, 4), np.uint16), (2, 2), 3)

        assert not zarr_utils.is_chunk_aligned_concat([a, b], axis=0)
        with pytest.raises(zarr_utils.NotChunkAlignedError):
            zarr_utils.virtual_concat([a, b], axis=0)


def test_virtual_stack_validates_compatibility():
    with tempfile.TemporaryDirectory() as tmpdir:
        a = _write_zarr(tmpdir, "a.zarr", np.ones((4, 4), np.uint16), (2, 2), 3)
        b = _write_zarr(tmpdir, "b.zarr", np.ones((4, 4), np.uint16), (4, 4), 3)
        with pytest.raises(ValueError):
            zarr_utils.virtual_stack([a, b])


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_virtual_transforms_compose(zarr_format):
    # A virtual array may itself be a source of another virtual array.
    with tempfile.TemporaryDirectory() as tmpdir:
        a = _write_zarr(
            tmpdir, "a.zarr", np.full((4, 4), 1, np.uint16), (2, 2), zarr_format
        )
        b = _write_zarr(
            tmpdir, "b.zarr", np.full((4, 4), 2, np.uint16), (2, 2), zarr_format
        )
        ea = zarr_utils.virtual_expand_dims(a, 1)
        eb = zarr_utils.virtual_expand_dims(b, 1)

        stacked = zarr_utils.virtual_stack([ea, eb])
        assert stacked.shape == (2, 1, 4, 4)
        assert np.array_equal(stacked[0, 0], a[:])
        assert np.array_equal(stacked[1, 0], b[:])
