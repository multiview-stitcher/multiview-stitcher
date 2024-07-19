import tempfile
from pathlib import Path

import dask.array as da
import numpy as np
import pytest

from multiview_stitcher import (
    fusion,
    io,
    msi_utils,
    mv_graph,
    registration,
    sample_data,
)
from multiview_stitcher.io import METADATA_TRANSFORM_KEY


@pytest.mark.parametrize(
    "ndim, overlap, N_c, N_t, dtype",
    [
        (2, 1, 1, 3, np.uint16),  # single pixel overlap not supported
        (2, 5, 1, 3, np.uint16),
        (2, 5, 1, 3, np.uint8),
        # (2, 5, 2, 3, np.uint8), # sporadically fails, need to investigate
        # (3, 5, 2, 3, np.uint16),
        (3, 1, 1, 3, np.uint8),
        (3, 5, 1, 3, np.uint8),
    ],
)
def test_diversity_stitching(ndim, overlap, N_c, N_t, dtype):
    sims = sample_data.generate_tiled_dataset(
        ndim=ndim,
        N_t=N_t,
        N_c=N_c,
        tile_size=15,
        tiles_x=2,
        tiles_y=1,
        tiles_z=1,
        overlap=overlap,
        zoom=10,
        dtype=dtype,
    )

    msims = [
        msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims
    ]

    # Run registration
    if overlap > 1:
        registration.register(
            msims,
            reg_channel_index=0,
            transform_key=METADATA_TRANSFORM_KEY,
            new_transform_key="affine_registered",
        )
    else:
        with pytest.raises(mv_graph.NotEnoughOverlapError):
            registration.register(
                msims,
                reg_channel_index=0,
                transform_key=METADATA_TRANSFORM_KEY,
                new_transform_key="affine_registered",
            )
        return

    sims = [msi_utils.get_sim_from_msim(msim) for msim in msims]

    # Run fusion
    fused = fusion.fuse(
        sims,
        transform_key="affine_registered",
    )

    # ensure that channel labels and order remain unchanged
    assert (
        fused.coords["c"].values.tolist()
        == sims[0].coords["c"].values.tolist()
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath_tif = str(Path(tmpdir) / "test.tif")
        filepath_zarr = str(Path(tmpdir) / "test.zarr")

        fused.data = da.to_zarr(
            fused.data, filepath_zarr, overwrite=True, return_stored=True
        )

        io.save_sim_as_tif(filepath_tif, fused)
