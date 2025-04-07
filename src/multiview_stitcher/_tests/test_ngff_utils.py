import json
import os
import shutil
import tempfile

import ngff_zarr
import numpy as np
import pytest

from multiview_stitcher import (
    io,
    msi_utils,
    ngff_utils,
    sample_data,
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
        ngff_utils.write_sim_to_ome_zarr(sim, zarr_path, overwrite=False)

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
def test_ome_zarr_read_write(ndim, N_t, N_c):
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

        sim_read = ngff_utils.read_sim_from_ome_zarr(
            zarr_path
        )  # , resolution_level=0)

        # check dims and channel names are the same
        # assert np.equal(sim.data, sim_read.data).all()
        assert np.array_equal(sim.dims, sim_read.dims)
        # TODO: consider restricting channel coords to string type
        assert np.array_equal(
            [str(v) for v in sim.coords["c"].values],
            [str(v) for v in sim_read.coords["c"].values],
        )


def test_multiscales_completion():
    sim = sample_data.generate_tiled_dataset(
        ndim=2,
        overlap=0,
        N_c=1,
        N_t=1,
        tile_size=30,
        tiles_x=2,
        tiles_y=1,
        tiles_z=1,
        spacing_x=0.1,
        spacing_y=0.1,
        spacing_z=2,
    )[1]

    with tempfile.TemporaryDirectory() as zarr_path:
        # write sim to ome zarr
        sim = ngff_utils.write_sim_to_ome_zarr(sim, zarr_path)

        # remove level 1 on disk
        shutil.rmtree(
            os.path.join(zarr_path, "1"),
            # ignore_errors=True,
        )

        # write again
        sim = ngff_utils.write_sim_to_ome_zarr(sim, zarr_path)

        with open(os.path.join(zarr_path, ".zattrs")) as f:
            json.load(f)

        # check that level 1 is now present
        sim_read = ngff_utils.read_sim_from_ome_zarr(
            zarr_path, resolution_level=1
        )

        # check dims and channel names are the same
        # assert np.equal(sim.data, sim_read.data).all()
        assert np.array_equal(sim.dims, sim_read.dims)
        # TODO: consider restricting channel coords to string type
        assert np.array_equal(
            [str(v) for v in sim.coords["c"].values],
            [str(v) for v in sim_read.coords["c"].values],
        )


def test_multiscales_overwrite():
    sim1 = si_utils.get_sim_from_array(
        np.zeros((40, 40)),
        translation={"y": 0, "x": 0},
    )
    sim2 = si_utils.get_sim_from_array(
        np.ones((40, 40)),
        translation={"y": 1, "x": 1},
    )

    with tempfile.TemporaryDirectory() as zarr_path:
        # write sim to ome zarr
        ngff_utils.write_sim_to_ome_zarr(sim1, zarr_path)

        # write again
        ngff_utils.write_sim_to_ome_zarr(sim2, zarr_path, overwrite=True)

        # check that read sim is equal to sim2 at
        # all resolution levels
        for res_level in range(2):
            sim_read = ngff_utils.read_sim_from_ome_zarr(
                zarr_path, resolution_level=res_level
            )
            assert np.min(sim_read.data) == 1
            assert np.max(sim_read.data) == 1

            for dim in sim_read.dims:
                if dim not in si_utils.SPATIAL_DIMS:
                    continue
                assert sim_read.coords[dim].values[0] > 0
