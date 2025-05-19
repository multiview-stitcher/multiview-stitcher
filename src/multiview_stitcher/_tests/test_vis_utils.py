import os
import tempfile

import numpy as np
import pytest
from matplotlib import pyplot as plt

import multiview_stitcher.spatial_image_utils as si_utils
from multiview_stitcher import (
    io,
    msi_utils,
    ngff_utils,
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

    indexed_sims1 = set(sims[0].coords.xindexes.dims)

    fig, ax = vis_utils.plot_positions(
        sims, transform_key=METADATA_TRANSFORM_KEY
    )

    fig, ax = vis_utils.plot_positions(
        msims, transform_key=METADATA_TRANSFORM_KEY
    )

    assert len(ax.collections) == len(msims)

    indexed_sims2 = set(sims[0].coords.xindexes.dims)

    assert indexed_sims1 == indexed_sims2


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


@pytest.mark.parametrize(
    "ndim, N_t, N_c, option",
    [
        (2, 1, 1, ""),
        (2, 2, 1, ""),
        (3, 1, 2, ""),
        (2, None, None, ""),
        (2, None, 1, "different_c_coords"),
    ],
)
def test_ome_zarr_ng(ndim, N_t, N_c, option):
    sims = sample_data.generate_tiled_dataset(
        ndim=ndim,
        overlap=0,
        N_c=N_c if N_c is not None else 1,
        N_t=N_t if N_t is not None else 1,
        tile_size=10,
        tiles_x=2,
        tiles_y=1,
        tiles_z=1,
        spacing_x=0.1,
        spacing_y=0.1,
        spacing_z=2,
    )

    # make sure to also test for the absence of c and t
    if N_c is None:
        for isim in range(len(sims)):
            sims[isim] = sims[isim].drop_vars("c")

    if N_t is None:
        for isim in range(len(sims)):
            sims[isim] = sims[isim].drop_vars("t")

    # test case of different
    if option == "different_c_coords":
        for isim in range(len(sims)):
            sims[isim] = sims[isim].assign_coords(c=[f"channel {isim}"])

    with tempfile.TemporaryDirectory() as data_dir:
        zarr_paths = [
            os.path.join(data_dir, f"sim_{isim}.zarr")
            for isim in range(len(sims))
        ]
        [
            ngff_utils.write_sim_to_ome_zarr(sim, zarr_paths[isim])
            for isim, sim in enumerate(sims)
        ]

        for single_layer in [True, False]:
            ng_json = vis_utils.generate_neuroglancer_json(
                ome_zarr_paths=zarr_paths,
                ome_zarr_urls=[
                    f"https://localhost:8000/{os.path.basename(zp)}"
                    for zp in zarr_paths
                ],
                sims=sims,
                transform_key=io.METADATA_TRANSFORM_KEY,
                single_layer=single_layer,
            )
            assert len(ng_json.keys())

        # test with channel coord
        if option != "different_c_coords":
            ng_json = vis_utils.generate_neuroglancer_json(
                ome_zarr_paths=zarr_paths,
                ome_zarr_urls=[
                    f"https://localhost:8000/{os.path.basename(zp)}"
                    for zp in zarr_paths
                ],
                sims=sims,
                channel_coord=sims[0].coords["c"].values[0],
                transform_key=io.METADATA_TRANSFORM_KEY,
                single_layer=single_layer,
            )
            assert len(ng_json.keys())

        # without sims
        ng_json = vis_utils.generate_neuroglancer_json(
            ome_zarr_paths=zarr_paths,
            ome_zarr_urls=[
                f"https://localhost:8000/{os.path.basename(zp)}"
                for zp in zarr_paths
            ],
        )
        assert len(ng_json.keys())
