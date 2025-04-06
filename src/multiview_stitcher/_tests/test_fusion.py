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
    sims = io.read_mosaic_into_sims(sample_data.get_mosaic_sample_data_path())

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
    fusedc = fused.data.compute(scheduler="single-threaded")

    assert np.min(fusedc) > 0


def test_fused_field_slice():
    """
    Make sure that slice in fused output
      - is properly aligned with the input
      - only requires input data from the equivalent input slice
    """

    # construct array that will complain if requested for
    # chunks that don't have z index 1

    def provide_only_slice(x, imval, block_info=None):
        block_slices = block_info[None]["array-location"]
        if block_slices[0][0] != 1:
            raise ValueError(
                "This part of the input array shouldn't be required"
            )
        else:
            return np.ones(x.shape) * imval

    imval = 1.0
    dim = da.map_blocks(
        provide_only_slice,
        da.empty((5, 50, 100), chunks=(1, 50, 50)),
        dtype=np.float32,
        imval=imval,
    )

    sdims = ["z", "y", "x"]
    spacing = {"z": 3.5, "y": 2.5, "x": 4.5}
    affine_translation = {"z": 1.3, "y": 1, "x": 2}
    sim = si_utils.get_sim_from_array(
        dim,
        dims=sdims,
        scale=spacing,
        transform_key=METADATA_TRANSFORM_KEY,
        affine=param_utils.affine_from_translation(
            [affine_translation[dim] for dim in sdims]
        ),
    )

    output_stack_properties = {
        "spacing": spacing,
        "origin": {
            dim: t + 1 * spacing[dim] for dim, t in affine_translation.items()
        },
        "shape": {"z": 1, "y": 40, "x": 70},
    }

    fused = fusion.fuse(
        [sim],
        transform_key=METADATA_TRANSFORM_KEY,
        interpolation_order=1,
        output_stack_properties=output_stack_properties,
    ).compute()

    assert not any(fused.data.flatten() - imval)


def test_3D_single_plane_fusion():
    """
    Make sure that 3D single plane fusion works
    (i.e. the z axis of the input has length 1)
    """
    sim = si_utils.get_sim_from_array(
        np.ones((1, 10, 10)),
        dims=["z", "y", "x"],
        transform_key=METADATA_TRANSFORM_KEY,
    )

    # fails if output_chunksize[z] != 1 because the
    # weight calculation assumes shape > 1
    fusion.fuse(
        [sim],
        output_shape={"z": 2, "y": 10, "x": 10},
        output_chunksize={"z": 1, "y": 10, "x": 10},
        transform_key=METADATA_TRANSFORM_KEY,
    ).compute(scheduler="single-threaded")


def test_blending_widths():
    """
    Simple test to check that the blending widths are taken into account
    """
    sims = io.read_mosaic_into_sims(sample_data.get_mosaic_sample_data_path())

    fused_small_bw = (
        fusion.fuse(
            sims,
            transform_key=METADATA_TRANSFORM_KEY,
            blending_widths={dim: 0.001 for dim in ["y", "x"]},
        )
        .compute()
        .data
    )

    fused_large_bw = (
        fusion.fuse(
            sims,
            transform_key=METADATA_TRANSFORM_KEY,
            blending_widths={dim: 10 for dim in ["y", "x"]},
        )
        .compute()
        .data
    )

    # make sure the fusion results are different
    assert not np.allclose(fused_small_bw, fused_large_bw)


def test_large_shape_fusion():
    """
    Make sure that arrays with shape > uin16 limit can be fused
    """
    sims = [
        si_utils.get_sim_from_array(
            np.ones((2, 50000)),
            dims=["y", "x"],
            transform_key=METADATA_TRANSFORM_KEY,
        )
        for _ in range(2)
    ]

    sims[1] = sims[1].assign_coords(x=sims[1].coords["x"] + 50000)

    # fails if output_chunksize[z] != 1 because the
    # weight calculation assumes shape > 1
    fused = fusion.fuse(
        sims,
        transform_key=METADATA_TRANSFORM_KEY,
    )

    assert fused.data.shape[-1] > 60000


@pytest.mark.parametrize(
    "input_chunksize",
    [
        {"y": 5, "x": 5},
        {"z": 4, "y": 5, "x": 5},
        {"z": 1, "y": 5, "x": 5},
        {"y": None, "x": None},  # numpy input
    ],
)
def test_fusion_chunksizes(input_chunksize):
    ndim = len(input_chunksize)
    output_chunksize = {
        dim: cs * 2 if cs is not None else 5
        for dim, cs in input_chunksize.items()
    }

    sims = [
        si_utils.get_sim_from_array(
            da.zeros(
                [2] + [10] * len(input_chunksize),
                chunks=[1] + list(input_chunksize.values()),
            )
            if input_chunksize["x"] is not None
            else np.zeros([2] + [10] * ndim),
            dims=["c"] + list(input_chunksize.keys()),
        )
        for _ in range(2)
    ]

    for set_output_chunksize in [True, False]:
        fused = fusion.fuse(
            sims,
            transform_key=METADATA_TRANSFORM_KEY,
            output_chunksize=output_chunksize
            if set_output_chunksize
            else None,
        )

        if set_output_chunksize:
            expected_chunksize = output_chunksize
        else:
            if input_chunksize["x"] is not None:
                expected_chunksize = input_chunksize
            else:
                expected_chunksize = {
                    dim: min(
                        fused.shape[-ndim + idim],
                        si_utils.get_default_spatial_chunksizes(ndim)[dim],
                    )
                    for idim, dim in enumerate(si_utils.SPATIAL_DIMS[-ndim:])
                }
        assert all(
            fused.data.chunksize[-ndim + idim] == expected_chunksize[dim]
            for idim, dim in enumerate(si_utils.SPATIAL_DIMS[-ndim:])
        )
