import tempfile

import numpy as np

from multiview_stitcher import msi_utils, param_utils
from multiview_stitcher import spatial_image_utils as si_utils


def test_complete_msim_zarr():
    msim = msi_utils.get_msim_from_sim(
        si_utils.get_sim_from_array(np.ones((10, 10)))
    )
    affine = param_utils.affine_to_xaffine(param_utils.random_affine())

    with tempfile.TemporaryDirectory() as tmpdirname:
        msi_utils.multiscale_spatial_image_to_zarr(msim, tmpdirname)
        before_completion = msi_utils.multiscale_spatial_image_from_zarr(
            tmpdirname
        )
        msi_utils.set_affine_transform(msim, affine, "test_key")
        msi_utils.complete_msim_zarr(msim, tmpdirname)
        after_completion = msi_utils.multiscale_spatial_image_from_zarr(
            tmpdirname
        )

    assert "test_key" not in before_completion["scale0"].data_vars
    assert "test_key" in after_completion["scale0"].data_vars


def test_update_msim_metadata_zarr():
    msim = msi_utils.get_msim_from_sim(
        si_utils.get_sim_from_array(np.ones((10, 10)))
    )
    affine1 = param_utils.affine_to_xaffine(param_utils.random_affine())
    affine2 = param_utils.affine_to_xaffine(param_utils.random_affine())

    msi_utils.set_affine_transform(msim, affine1, "test_key")

    with tempfile.TemporaryDirectory() as tmpdirname:
        msi_utils.multiscale_spatial_image_to_zarr(msim, tmpdirname)
        before_update = msi_utils.multiscale_spatial_image_from_zarr(
            tmpdirname
        )
        msi_utils.set_affine_transform(msim, affine2, "test_key")
        msi_utils.update_msim_metadata_zarr(msim, tmpdirname)
        after_update = msi_utils.multiscale_spatial_image_from_zarr(tmpdirname)

    ti = msi_utils.get_transform_from_msim(before_update, "test_key")
    tf = msi_utils.get_transform_from_msim(after_update, "test_key")

    assert not np.allclose(ti, tf)
