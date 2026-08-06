"""What multiview-stitcher does under each zarr-python version.

Every test here runs under both zarr-python v2 and v3 and asserts the branch
that belongs to the installed one, so neither environment is left with a test
that cannot fail. See :mod:`multiview_stitcher._tests._zarr_marks` for the
division of labour between the two.
"""

import os

import dask.array as da
import numpy as np
import pytest
import tifffile
import zarr

from multiview_stitcher import _zarr_compat, io, ngff_utils, zarr_utils
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher._zarr_compat import ZARR_V3, ZarrV3Required


def _zarray(path, data, chunks):
    zarray = zarr.open_array(
        str(path), mode="w", shape=data.shape, chunks=chunks, dtype=data.dtype
    )
    zarray[:] = data
    return zarray


def test_zarr_v3_flag_agrees_with_the_installed_library():
    assert zarr.__version__.startswith("3.") == ZARR_V3


@pytest.mark.parametrize(
    "version, expected",
    [("2.18.7", 2), ("3.0.0b2", 3), ("3.2.1", 3), ("10.0.0", 10)],
)
def test_version_is_parsed_not_string_compared(monkeypatch, version, expected):
    """``"10.0" >= "3"`` is False lexicographically, so parse the number."""
    monkeypatch.setattr(zarr, "__version__", version)
    assert _zarr_compat._zarr_major_version() == expected


def test_virtual_array_support_tracks_the_library_version():
    assert zarr_utils.supports_virtual_arrays() == ZARR_V3


def test_virtual_transforms_are_reported_unavailable_under_zarr_v2(tmp_path):
    """The ``is_*`` predicates answer False rather than raising, because their
    callers use them to choose between the lazy and the eager path."""
    data = np.arange(16, dtype=np.uint16).reshape(4, 4)
    zarrays = [
        _zarray(tmp_path / "a.zarr", data, (2, 2)),
        _zarray(tmp_path / "b.zarr", data, (2, 2)),
    ]

    assert zarr_utils.is_stackable(zarrays) == ZARR_V3
    assert zarr_utils.is_chunk_aligned_concatenate(zarrays, 0) == ZARR_V3


@pytest.mark.skipif(ZARR_V3, reason="zarr-python v2 behaviour")
def test_virtual_transforms_name_the_reason_when_called_under_zarr_v2(
    tmp_path,
):
    """Callers that skip the predicates get an explanation, not an
    ``AttributeError`` from inside zarr."""
    data = np.arange(16, dtype=np.uint16).reshape(4, 4)
    zarray = _zarray(tmp_path / "a.zarr", data, (2, 2))

    for call in (
        lambda: zarr_utils.expand_dims(zarray, 2),
        lambda: zarr_utils.stack([zarray, zarray]),
        lambda: zarr_utils.concatenate([zarray, zarray], 0),
    ):
        with pytest.raises(ZarrV3Required, match="zarr-python >= 3"):
            call()


def test_sim_from_a_zarr_array_falls_back_to_dask_under_zarr_v2(tmp_path):
    """A zarr array is still valid input under v2 - it is read through dask
    instead of staying zarr-backed, which costs the byte-passthrough fast
    paths but returns the same image."""
    data = np.arange(64, dtype=np.uint16).reshape(8, 8)
    zarray = _zarray(tmp_path / "input.zarr", data, (4, 4))

    def build():
        return si_utils.get_sim_from_array(
            zarray,
            dims=["y", "x"],
            scale={"y": 1.0, "x": 1.0},
            translation={"y": 0.0, "x": 0.0},
        )

    if ZARR_V3:
        sim = build()
        assert si_utils.is_xarray_zarr_backed(sim)
    else:
        with pytest.warns(RuntimeWarning, match="zarr-python >= 3"):
            sim = build()
        assert not si_utils.is_xarray_zarr_backed(sim)
        assert isinstance(sim.data, da.Array)

    # Either way the singleton t/c axes are added and the pixels survive.
    assert list(sim.dims) == ["t", "c", "y", "x"]
    np.testing.assert_array_equal(np.asarray(sim.isel(t=0, c=0).data), data)


def test_tif_zarr_backend_falls_back_to_dask_under_zarr_v2(tmp_path):
    data = np.arange(2 * 4 * 5, dtype=np.uint16).reshape(2, 4, 5)
    filepath = str(tmp_path / "stack.tif")
    tifffile.imwrite(filepath, data, metadata={"axes": "CYX"})

    def build():
        return io.read_tiff_into_spatial_xarray(
            filepath, dims=["c", "y", "x"], array_backend="zarr"
        )

    if ZARR_V3:
        sim = build()
    else:
        with pytest.warns(RuntimeWarning, match="zarr-python >= 3"):
            sim = build()
        assert isinstance(sim.data, da.Array)

    np.testing.assert_array_equal(
        np.asarray(sim.isel(t=0).data), data
    )


def test_ome_zarr_0_5_is_refused_under_zarr_v2():
    """0.5 is a zarr v3 hierarchy, so v2 cannot write it at all - unlike the
    cases above there is no fallback, and the caller has to hear why."""
    if ZARR_V3:
        assert ngff_utils.zarr_group_creation_kwargs_for_ngff_version("0.5") == {
            "zarr_format": 3
        }
        return

    with pytest.raises(ZarrV3Required, match="OME-Zarr 0.5"):
        ngff_utils.zarr_group_creation_kwargs_for_ngff_version("0.5")

    with pytest.raises(ZarrV3Required, match="OME-Zarr 0.5"):
        ngff_utils.update_zarr_array_creation_kwargs_for_ngff_version(
            "0.5", None
        )


@pytest.mark.parametrize("n_batch", [None, 1, 2])
def test_empty_chunks_are_written_under_both_zarr_versions(tmp_path, n_batch):
    """Resolution levels are written block by block, so a block holding only
    the fill value must still land on disk - readers otherwise see a hole
    where a chunk should be.

    The two libraries request this differently: v2 per array handle, v3 through
    its global config, because a v3 ``config=`` is dropped when the array being
    opened already exists. ``n_batch=None`` writes through dask, the others
    block by block.
    """
    zarr_path = str(tmp_path / "zeros.ome.zarr")
    sim = si_utils.get_sim_from_array(
        np.zeros((8, 8), dtype=np.uint16),
        dims=["y", "x"],
        scale={"y": 1.0, "x": 1.0},
        translation={"y": 0.0, "x": 0.0},
    ).chunk({"t": 1, "c": 1, "y": 4, "x": 4})

    ngff_utils.write_sim_to_ome_zarr(
        sim,
        zarr_path,
        ngff_version="0.4",
        show_progressbar=False,
        batch_options={"n_batch": n_batch},
    )

    level0 = os.path.join(zarr_path, "0")
    chunk_files = [
        os.path.join(root, name)
        for root, _, names in os.walk(level0)
        for name in names
        if not name.startswith(".")
    ]
    # 1 t * 1 c * 2 y * 2 x chunks, none of them elided for being all zeros.
    assert len(chunk_files) == 4
