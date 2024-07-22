import os
import tempfile
import warnings

import dask.array as da
import numpy as np
import pytest
import xarray as xr

import multiview_stitcher.spatial_image_utils as si_utils
from multiview_stitcher import (
    fusion,
    io,
    msi_utils,
    param_utils,
    sample_data,
    weights,
)
from multiview_stitcher.io import METADATA_TRANSFORM_KEY


def test_fuse_sims():
    sims = io.read_mosaic_image_into_list_of_spatial_xarrays(
        sample_data.get_mosaic_sample_data_path()
    )

    # suppress pandas future warning occuring within xarray.concat
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)

        # test with two channels
        for isim, sim in enumerate(sims):
            sims[isim] = xr.concat([sim] * 2, dim="c").assign_coords(
                c=[sim.coords["c"].data[0], sim.coords["c"].data[0] + "_2"]
            )

    xfused = fusion.fuse(
        sims,
        transform_key=METADATA_TRANSFORM_KEY,
    )

    # check output is dask array and hasn't been converted into numpy array
    assert type(xfused.data) == da.core.Array
    assert xfused.dtype == sims[0].dtype

    # xfused.compute()
    xfused = xfused.compute(scheduler="single-threaded")

    assert xfused.dtype == sims[0].dtype
    assert METADATA_TRANSFORM_KEY in si_utils.get_tranform_keys_from_sim(
        xfused
    )


@pytest.mark.parametrize(
    "ndim, weights_func",
    [
        (2, None),
        (2, weights.content_based),
        (3, None),
        (3, weights.content_based),
    ],
)
def test_multi_view_fusion(ndim, weights_func):
    nviews = 3

    sims = [
        sample_data.generate_tiled_dataset(
            ndim=ndim,
            overlap=0,
            N_c=1,
            N_t=1,
            tile_size=20,
            tiles_x=1,
            tiles_y=1,
            tiles_z=1,
            spacing_x=1,
            spacing_y=1,
            spacing_z=1,
        )[0]
        for _ in range(nviews)
    ]

    # prepare assertion
    for _, sim in enumerate(sims):
        sim.data += 1

    fused = fusion.fuse(
        sims[:],
        transform_key=METADATA_TRANSFORM_KEY,
        weights_func=weights_func,
        weights_func_kwargs={"sigma_1": 1, "sigma_2": 2}
        if weights_func == weights.content_based
        else None,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        fused.data = da.to_zarr(
            fused.data,
            os.path.join(tmpdir, "fused_sim.zarr"),
            overwrite=True,
            return_stored=True,
            compute=True,
        )

        mfused = msi_utils.get_msim_from_sim(fused, scale_factors=[])

        fused_path = os.path.join(tmpdir, "fused.zarr")
        mfused.to_zarr(fused_path, mode="w")

        mfused = msi_utils.multiscale_spatial_image_from_zarr(fused_path)

        assert fused.data.min() > 0


def test_fused_field_coverage():
    scale = {"y": 2, "x": 0.5}
    affine = param_utils.affine_from_translation([1000, -1000])

    sims = []
    N_x, N_y = 3, 3
    for ix in range(N_x):
        for iy in range(N_y):
            sims.append(
                si_utils.get_sim_from_array(
                    da.ones((20, 20), chunks=(10, 10)) + ix + iy,
                    dims=["y", "x"],
                    scale=scale,
                    translation={
                        "y": iy * 20 * scale["y"] - 100,
                        "x": ix * 20 * scale["x"] + 100,
                    },
                    affine=affine,
                    transform_key=METADATA_TRANSFORM_KEY,
                )
            )

    fused = fusion.fuse(
        sims,
        transform_key=METADATA_TRANSFORM_KEY,
        output_chunksize=13,
        output_spacing=scale,
    )

    assert np.min(fused.data.compute()) > 0
