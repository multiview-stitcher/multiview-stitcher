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
        print(tmpdirname)
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
