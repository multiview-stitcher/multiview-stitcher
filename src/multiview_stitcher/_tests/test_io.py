import os
import tempfile

import numpy as np
import pytest

from multiview_stitcher import io, sample_data


@pytest.mark.parametrize(
    "ndim, N_t, N_c",
    [(ndim, N_t, N_c) for ndim in [2, 3] for N_t in [1, 2] for N_c in [1, 2]],
)
def test_tiff_io(ndim, N_t, N_c):
    """
    Could be much more general
    """

    tile_size = 10
    spacing_x = 0.5
    spacing_y = 0.5
    spacing_z = 0.5
    sims = sample_data.generate_tiled_dataset(
        ndim=ndim,
        overlap=0,
        N_c=N_c,
        N_t=N_t,
        tile_size=tile_size,
        tiles_x=1,
        tiles_y=1,
        tiles_z=1,
        spacing_x=spacing_x,
        spacing_y=spacing_y,
        spacing_z=spacing_z,
        drift_scale=0,
        shift_scale=0,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.tif")
        io.save_sim_as_tif(filepath, sims[0])

        sims_io = io.read_tiff_into_spatial_xarray(
            filepath, channel_names=["ch%s" % i for i in range(N_c)]
        )

        assert sims[0].data.ndim == sims_io.data.ndim

        # check that all dims have the same length
        for dim in sims[0].dims:
            assert len(sims[0].coords[dim]) == len(sims_io.coords[dim])
            # assert np.allclose(sims[0].coords[dim], sims_io.coords[dim])

        # check image values are the same
        # ignore coordinates for this test
        for dim in sims[0].dims:
            sims[0].coords[dim] = np.arange(len(sims[0].coords[dim]))
            sims_io.coords[dim] = np.arange(len(sims_io.coords[dim]))

        assert (sims[0] == sims_io).min()
