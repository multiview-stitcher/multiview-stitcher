import numpy as np

from multiview_stitcher import param_utils
from multiview_stitcher import spatial_image_utils as si_utils


def test_rebase_param_order():
    p1 = param_utils.identity_transform(2)
    sim = si_utils.get_sim_from_array(
        np.zeros([10] * 3), affine=p1, transform_key="p1"
    )

    p2 = param_utils.affine_to_xaffine(
        param_utils.affine_from_translation([2, 3, 0])
    )

    si_utils.set_sim_affine(
        sim, p2, transform_key="p2", base_transform_key="p1"
    )

    assert np.all(
        p2.coords["x_in"].data
        == si_utils.get_affine_from_sim(sim, "p2").coords["x_in"].data
    )
