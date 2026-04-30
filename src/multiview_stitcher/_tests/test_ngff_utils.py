import os
import shutil
import tempfile

import ngff_zarr
import numpy as np
import pytest
import zarr

from multiview_stitcher import (
    io,
    msi_utils,
    ngff_utils,
    sample_data,
)
from multiview_stitcher import spatial_image_utils as si_utils


@pytest.mark.parametrize(
    "ndim, ngff_version, n_batch",
    [(ndim, ngff_version, n_batch)
    for ndim in (2, 3)
    for ngff_version in ("0.4", "0.5")
    for n_batch in (None, 2)
    ],
)
def test_round_trip(ndim, ngff_version, n_batch):
    """Round-trip a sim and msim through OME-Zarr and verify pixel data and
    spatial coordinates are preserved at every resolution level."""
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

    if zarr.__version__ < "3" and ngff_version >= "0.5":
        pytest.skip("zarr>=3 required for ngff_version 0.5")

    # sim
    sdims = si_utils.get_spatial_dims_from_sim(sim)

    with tempfile.TemporaryDirectory() as zarr_path:
        ngff_utils.write_sim_to_ome_zarr(
            sim,
            zarr_path,
            overwrite=False,
            ngff_version=ngff_version,
            batch_options={
                "n_batch": n_batch,
            },
        )

        sim_read = ngff_utils.read_sim_from_ome_zarr(zarr_path)

        for dim in sdims:
            assert np.allclose(
                sim.coords[dim].values, sim_read.coords[dim].values
            )

        assert np.allclose(sim.data, sim_read.data)

    # msim
    scale_factors = [2, 2]
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

        assert np.allclose(
            msim[f"scale{len(scale_factors)}/image"].data,
            msim_read[f"scale{len(scale_factors)}/image"].data,
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


@pytest.mark.parametrize(
    "ndim, N_t, N_c",
    [(2, 1, 1), (2, 2, 1), (3, 1, 2), (2, None, None)],
)
def test_ome_zarr_read_write(ndim, N_t, N_c):
    """Write a sim to OME-Zarr and read it back, checking that dims, channel
    names and omero window metadata are preserved for various t/c combinations."""
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

        metadata = zarr.open_group(zarr_path).attrs.asdict()

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


def test_read_msim_from_ome_zarr():
    """Verify that read_msim_from_ome_zarr returns a multiscale image with
    correct pixel data, channel names and more than one resolution level."""
    sim = sample_data.generate_tiled_dataset(
        ndim=2,
        overlap=0,
        N_c=2,
        N_t=1,
        tile_size=202,
        tiles_x=2,
        tiles_y=1,
        tiles_z=1,
        spacing_x=0.1,
        spacing_y=0.1,
        spacing_z=2,
        random_data=True,
    )[1]

    with tempfile.TemporaryDirectory() as zarr_path:
        sim = ngff_utils.write_sim_to_ome_zarr(sim, zarr_path)

        msim_read = ngff_utils.read_msim_from_ome_zarr(zarr_path)
        sim_read = msi_utils.get_sim_from_msim(msim_read, scale="scale0")

        assert np.array_equal(sim.dims, sim_read.dims)
        assert np.allclose(sim.data, sim_read.data)
        assert np.array_equal(
            [str(v) for v in sim.coords["c"].values],
            [str(v) for v in sim_read.coords["c"].values],
        )

        scale_keys = msi_utils.get_sorted_scale_keys(msim_read)
        assert len(scale_keys) > 1
        for scale_key in scale_keys:
            sim_scale = msi_utils.get_sim_from_msim(
                msim_read, scale=scale_key
            )
            assert np.array_equal(
                [str(v) for v in sim.coords["c"].values],
                [str(v) for v in sim_scale.coords["c"].values],
            )


def test_multiscales_completion():
    """Check that writing without overwrite completes a partially deleted pyramid:
    after removing a resolution level on disk, re-writing fills it in and the
    metadata remains valid."""
    sim = sample_data.generate_tiled_dataset(
        ndim=2,
        overlap=0,
        N_c=1,
        N_t=1,
        tile_size=202,
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
        )

        # write again
        sim = ngff_utils.write_sim_to_ome_zarr(sim, zarr_path)

        # check that metadata is present
        zarr.open_group(zarr_path, mode="r").attrs.asdict()

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
    """Verify that overwrite=True replaces both pixel data and spatial
    coordinates at all resolution levels with those of the new sim."""
    sim1 = si_utils.get_sim_from_array(
        np.zeros((202, 202)),
        translation={"y": 0, "x": 0},
    )
    sim2 = si_utils.get_sim_from_array(
        np.ones((202, 202)),
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
                zarr_path, resolution_level=res_level)
            assert np.min(sim_read.data) == 1
            assert np.max(sim_read.data) == 1

            for dim in sim_read.dims:
                if dim not in si_utils.SPATIAL_DIMS:
                    continue
                assert sim_read.coords[dim].values[0] > 0


@pytest.mark.parametrize("ngff_version", ["0.4", "0.5"])
def test_update_ome_zarr_multiscales_metadata(ngff_version):
    """Write an OME-Zarr with one origin, call update_ome_zarr_multiscales_metadata
    with a new origin, then read back and assert the scale0 translation was
    updated while the multiscales key structure is intact."""
    if zarr.__version__ < "3" and ngff_version >= "0.5":
        pytest.skip("zarr>=3 required for ngff_version 0.5")

    spacing = {"y": 0.5, "x": 0.5}
    translation_orig = {"y": 10.0, "x": 20.0}
    translation_new = {"y": 3.0, "x": 7.0}

    sim = si_utils.get_sim_from_array(
        np.zeros((202, 202)),
        scale=spacing,
        translation=translation_orig,
    )

    with tempfile.TemporaryDirectory() as zarr_path:
        ngff_utils.write_sim_to_ome_zarr(
            sim, zarr_path, ngff_version=ngff_version
        )

        # build an updated msim with different translation
        sim_new = si_utils.get_sim_from_array(
            np.zeros((202, 202)),
            scale=spacing,
            translation=translation_new,
        )
        msim_new = msi_utils.get_msim_from_sim(sim_new, scale_factors=[])
        n_levels = len(
            msi_utils.get_sorted_scale_keys(
                ngff_utils.read_msim_from_ome_zarr(zarr_path)
            )
        )
        # match the number of resolution levels on disk
        msim_new = msi_utils.get_msim_from_sim(
            sim_new,
            scale_factors=[2] * (n_levels - 1),
        )

        ngff_utils.update_ome_zarr_multiscales_metadata(
            zarr_path, msim_new, transform_key=None
        )

        # read back and verify scale0 translation was updated
        sim_read = ngff_utils.read_sim_from_ome_zarr(zarr_path, resolution_level=0)
        sdims = si_utils.get_spatial_dims_from_sim(sim_read)
        for dim in sdims:
            assert np.isclose(
                sim_read.coords[dim].values[0],
                translation_new[dim],
            ), f"Expected {translation_new[dim]} for dim {dim}, got {sim_read.coords[dim].values[0]}"

        # verify that omero metadata (if present) is still intact
        root = zarr.open_group(zarr_path, mode="r")
        all_attrs = dict(root.attrs)
        if ngff_version == "0.5":
            assert "multiscales" in all_attrs.get("ome", {})
        else:
            assert "multiscales" in all_attrs
