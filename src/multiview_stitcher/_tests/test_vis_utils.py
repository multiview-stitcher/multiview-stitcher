import numpy as np
import pytest
from matplotlib import pyplot as plt

import multiview_stitcher.spatial_image_utils as si_utils
from multiview_stitcher import (
    msi_utils,
    # io,
    sample_data,
    vis_utils,
)
from multiview_stitcher.io import METADATA_TRANSFORM_KEY


@pytest.mark.parametrize(
    "ndim, N_t",
    [(2, 1), (2, 2), (3, 1), (3, 2)],
)
def test_plot_positions(ndim, N_t, monkeypatch):
    sims = sample_data.generate_tiled_dataset(
        ndim=ndim,
        N_c=2,
        N_t=N_t,
        tile_size=5,
        tiles_x=2,
        tiles_y=2,
        tiles_z=2,
    )

    msims = [
        msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims
    ]

    monkeypatch.setattr(plt, "show", lambda: None)

    fig, ax = vis_utils.plot_positions(
        msims, transform_key=METADATA_TRANSFORM_KEY
    )

    assert len(ax.collections) == len(msims)


def test_plot_positions_single_coord(monkeypatch):
    sim = si_utils.get_sim_from_array(
        np.random.randint(0, 255, (1, 100, 100)),
        dims=["z", "y", "x"],
    )

    monkeypatch.setattr(plt, "show", lambda: None)

    vis_utils.plot_positions(
        [msi_utils.get_msim_from_sim(sim, scale_factors=[])],
        transform_key=si_utils.DEFAULT_TRANSFORM_KEY,
        use_positional_colors=False,
    )

    vis_utils.plot_positions(
        [msi_utils.get_msim_from_sim(sim, scale_factors=[])],
        transform_key=si_utils.DEFAULT_TRANSFORM_KEY,
        use_positional_colors=False,
        spacing={"x": 1, "y": 1, "z": 1},
    )
