import json
import os
import tempfile

import ngff_zarr
import numpy as np
import pytest

from multiview_stitcher import (
    io,
    msi_utils,
    ngff_utils,
    sample_data,
    vis_utils,
)
from multiview_stitcher import spatial_image_utils as si_utils


@pytest.mark.parametrize(
    "ndim",
    [2, 3],
)
def test_round_trip(ndim):
    sim = sample_data.generate_tiled_dataset(
        ndim=ndim,
        overlap=0,
        N_c=1,
        N_t=1,
        tile_size=30,
        tiles_x=1,
        tiles_y=2,
        tiles_z=1,
        spacing_x=1,
        spacing_y=1,
        spacing_z=1,
    )[1]

    # sim
    sdims = si_utils.get_spatial_dims_from_sim(sim)

    with tempfile.TemporaryDirectory() as zarr_path:
        ngff_utils.write_sim_to_ome_zarr(sim, zarr_path)
        sim_read = ngff_utils.read_sim_from_ome_zarr(zarr_path)

        for dim in sdims:
            assert np.allclose(
                sim.coords[dim].values, sim_read.coords[dim].values
            )

        assert np.allclose(sim.data, sim_read.data)

    # msim
    scale_factors = [2, 4]
    msim = msi_utils.get_msim_from_sim(sim, scale_factors=scale_factors)
    with tempfile.TemporaryDirectory() as zarr_path:
        ngff_multiscales = ngff_utils.msim_to_ngff_multiscales(
            msim, transform_key=io.METADATA_TRANSFORM_KEY
        )
        ngff_zarr.to_ngff_zarr(zarr_path, ngff_multiscales)

        msim_read = ngff_utils.ngff_multiscales_to_msim(
            ngff_zarr.from_ngff_zarr(zarr_path),
            transform_key=io.METADATA_TRANSFORM_KEY,
        )

    for ires in range(len(scale_factors) + 1):
        assert np.allclose(
            msi_utils.get_sim_from_msim(msim_read, scale=f"scale{ires}")
            .coords["y"]
            .values,
            msi_utils.get_sim_from_msim(msim, scale=f"scale{ires}")
            .coords["y"]
            .values,
        )

    assert len(msi_utils.get_sorted_scale_keys(msim)) == len(
        msi_utils.get_sorted_scale_keys(msim_read)
    )

    assert np.allclose(
        msim[f"scale{len(scale_factors)}/image"].data,
        msim_read[f"scale{len(scale_factors)}/image"].data,
    )


@pytest.mark.parametrize(
    "ndim, N_t, N_c",
    [(2, 1, 1), (2, 2, 1), (3, 1, 2), (2, None, None)],
)
def test_ome_zarr_ng(ndim, N_t, N_c):
    sim = sample_data.generate_tiled_dataset(
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
    )[1]

    # make sure to also test for the absence of c and t
    if N_c is None:
        sim = sim.drop_vars("c")

    if N_t is None:
        sim = sim.drop_vars("t")

    with tempfile.TemporaryDirectory() as zarr_path:
        sim = ngff_utils.write_sim_to_ome_zarr(sim, zarr_path)

        with open(os.path.join(zarr_path, ".zattrs")) as f:
            metadata = json.load(f)

        if N_c is not None:
            assert "omero" in metadata
            assert "window" in metadata["omero"]["channels"][0]

        for single_layer in [True, False]:
            ng_json = vis_utils.generate_neuroglancer_json(
                [sim],
                [zarr_path],
                [
                    f"https://localhost:8000/{os.path.basename(zp)}"
                    for zp in [zarr_path]
                ],
                # channel_coord=sim.coords["c"].values[0],
                transform_key=io.METADATA_TRANSFORM_KEY,
                single_layer=single_layer,
            )
            assert len(ng_json.keys())
