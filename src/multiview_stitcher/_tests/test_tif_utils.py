import os
import tempfile

import numpy as np
import pytest
import tifffile

from multiview_stitcher.tif_utils import (
    tif_to_dask_plane_chunks,
    tif_to_virtual_zarr_v3_plane_chunks,
)


def test_tif_to_dask_plane_chunks_single_page():
    data = np.arange(4 * 5, dtype=np.uint16).reshape(4, 5)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.tif")
        tifffile.imwrite(filepath, data)

        out = tif_to_dask_plane_chunks(filepath)

        assert out.shape == data.shape
        np.testing.assert_array_equal(out.compute(), data)


@pytest.mark.parametrize(
    "backend",
    ["dask", "zarr"],
)
def test_tif_multi_axis_plane_chunks(backend):
    """
    A TIFF with two non-spatial axes (Z and C) should be read back with
    those axes separate, not flattened into a single page axis.
    """
    n_z, n_c, n_y, n_x = 3, 2, 4, 5
    data = np.arange(n_z * n_c * n_y * n_x, dtype=np.uint16).reshape(
        n_z, n_c, n_y, n_x
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.tif")
        tifffile.imwrite(filepath, data, metadata={"axes": "ZCYX"})

        if backend == "dask":
            out = tif_to_dask_plane_chunks(filepath)
            out = np.asarray(out.compute())
        else:
            out, _store = tif_to_virtual_zarr_v3_plane_chunks(filepath)
            out = out[:]

        assert out.shape == data.shape
        np.testing.assert_array_equal(out, data)
