import tempfile

import numpy as np

from multiview_stitcher import msi_utils, param_utils
from multiview_stitcher import spatial_image_utils as si_utils


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
