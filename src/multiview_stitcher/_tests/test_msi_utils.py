import tempfile

import numpy as np
import pytest

from multiview_stitcher import msi_utils, param_utils
from multiview_stitcher import spatial_image_utils as si_utils


@pytest.mark.parametrize(
    "dims",
    [
        ["y", "x"],
        ["z", "y", "x"],
        ["c", "z", "y", "x"],
        ["t", "c", "y", "x"],
    ],
)
def test_calc_resolution_levels(dims):
    sim = si_utils.get_sim_from_array(np.ones((100,) * len(dims)), dims=dims)
    msim = msi_utils.get_msim_from_sim(sim)

    # assert that for each level at least one spatial dimension is downsampled
    previous_shape = si_utils.get_shape_from_sim(sim)
    for scale_key in msi_utils.get_sorted_scale_keys(msim)[1:]:
        current_shape = si_utils.get_shape_from_sim(msim[scale_key])
        downsampled = False
        for dim in si_utils.get_spatial_dims_from_sim(sim):
            if current_shape[dim] < previous_shape[dim]:
                downsampled = True
        assert downsampled
        previous_shape = current_shape

    scale_keys = msi_utils.get_sorted_scale_keys(msim)
    assert len(scale_keys) > 1


def test_update_msim_transforms_zarr():
    msim = msi_utils.get_msim_from_sim(
        si_utils.get_sim_from_array(np.ones((10, 10)))
    )
    affine1 = param_utils.affine_to_xaffine(param_utils.random_affine())
    affine2 = param_utils.affine_to_xaffine(param_utils.random_affine())

    msi_utils.set_affine_transform(msim, affine1, "test_key")

    # overwrite existing transform
    with tempfile.TemporaryDirectory() as tmpdirname:
        msi_utils.multiscale_spatial_image_to_zarr(msim, tmpdirname)
        before_update = msi_utils.multiscale_spatial_image_from_zarr(
            tmpdirname
        )
        msi_utils.set_affine_transform(msim, affine2, "test_key")
        msi_utils.update_msim_transforms_zarr(msim, tmpdirname, overwrite=True)
        after_update = msi_utils.multiscale_spatial_image_from_zarr(tmpdirname)

    ti = msi_utils.get_transform_from_msim(before_update, "test_key")
    tf = msi_utils.get_transform_from_msim(after_update, "test_key")

    # don't overwrite existing transform
    with tempfile.TemporaryDirectory() as tmpdirname:
        msi_utils.multiscale_spatial_image_to_zarr(msim, tmpdirname)
        before_update = msi_utils.multiscale_spatial_image_from_zarr(
            tmpdirname
        )
        msi_utils.set_affine_transform(msim, affine2, "test_key")
        msi_utils.update_msim_transforms_zarr(
            msim, tmpdirname, overwrite=False
        )
        after_update = msi_utils.multiscale_spatial_image_from_zarr(tmpdirname)

    ti = msi_utils.get_transform_from_msim(before_update, "test_key")
    tf = msi_utils.get_transform_from_msim(after_update, "test_key")

    assert np.allclose(ti, tf)

    # check that new transforms are added
    with tempfile.TemporaryDirectory() as tmpdirname:
        msi_utils.multiscale_spatial_image_to_zarr(msim, tmpdirname)
        before_update = msi_utils.multiscale_spatial_image_from_zarr(
            tmpdirname
        )
        msi_utils.set_affine_transform(msim, affine2, "test_key2")
        msi_utils.update_msim_transforms_zarr(
            msim, tmpdirname, overwrite=False
        )
        after_update = msi_utils.multiscale_spatial_image_from_zarr(tmpdirname)

    keys = msi_utils.get_transforms_from_dataset_as_dict(
        after_update["scale0"]
    )
    assert "test_key" in keys
    assert "test_key2" in keys
    assert "image" in after_update["scale0"].data_vars

    # check that existing transforms are kept
    with tempfile.TemporaryDirectory() as tmpdirname:
        msi_utils.multiscale_spatial_image_to_zarr(msim, tmpdirname)
        before_update = msi_utils.multiscale_spatial_image_from_zarr(
            tmpdirname
        )
        msim = msi_utils.get_msim_from_sim(
            si_utils.get_sim_from_array(np.ones((10, 10)))
        )
        msi_utils.update_msim_transforms_zarr(msim, tmpdirname, overwrite=True)
        after_update = msi_utils.multiscale_spatial_image_from_zarr(tmpdirname)

    keys = msi_utils.get_transforms_from_dataset_as_dict(
        after_update["scale0"]
    )
    assert "test_key" in keys
    assert "image" in after_update["scale0"].data_vars


def test_get_res_level_from_binning_factors():
    """Test that get_res_level_from_binning_factors returns correct resolution level."""
    # Create a test image with known multiscale structure
    # scale0: 100x100, scale1: 50x50 (factor 2), scale2: 25x25 (factor 4)
    sim = si_utils.get_sim_from_array(
        np.ones((100, 100)),
        dims=["y", "x"],
        scale={"y": 1.0, "x": 1.0},
        translation={"y": 0.0, "x": 0.0},
    )
    
    # Create multiscale with specific factors
    scale_factors = [
        {"y": 2, "x": 2},  # scale1: 50x50
        {"y": 2, "x": 2},  # scale2: 25x25
    ]
    msim = msi_utils.get_msim_from_sim(sim, scale_factors=scale_factors)
    
    # Test case 1: binning factors match scale1 exactly
    scale_key, remaining = msi_utils.get_res_level_from_binning_factors(
        msim, {"y": 2, "x": 2}
    )
    assert scale_key == "scale1"
    assert remaining == {"y": 1, "x": 1}
    
    # Test case 2: binning factors match scale2 exactly
    scale_key, remaining = msi_utils.get_res_level_from_binning_factors(
        msim, {"y": 4, "x": 4}
    )
    assert scale_key == "scale2"
    assert remaining == {"y": 1, "x": 1}
    
    # Test case 3: binning factors less than any scale (use scale0)
    scale_key, remaining = msi_utils.get_res_level_from_binning_factors(
        msim, {"y": 1, "x": 1}
    )
    assert scale_key == "scale0"
    assert remaining == {"y": 1, "x": 1}
    
    # Test case 4: binning factors require additional binning on top of scale1
    # We want binning of 4, and scale1 provides 2, so we need remaining 2
    scale_key, remaining = msi_utils.get_res_level_from_binning_factors(
        msim, {"y": 2, "x": 2}
    )
    assert scale_key == "scale1"
    # Remaining should be 1 since scale1 already gives us factor 2
    assert remaining == {"y": 1, "x": 1}

