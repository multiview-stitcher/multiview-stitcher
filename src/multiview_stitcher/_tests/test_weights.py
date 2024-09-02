import multiview_stitcher.spatial_image_utils as si_utils
from multiview_stitcher import (
    io,
    param_utils,
    sample_data,
    weights,
)


def test_blending_weights():
    """
    Test blending weights calculation.
    """

    sims = io.read_mosaic_image_into_list_of_spatial_xarrays(
        sample_data.get_mosaic_sample_data_path()
    )

    ndim = si_utils.get_ndim_from_sim(sims[0])

    stack_propss = [
        si_utils.get_stack_properties_from_sim(sim) for sim in sims
    ]

    affine = param_utils.identity_transform(ndim)

    weights.get_blending_weights(
        stack_propss[0],
        stack_propss[1],
        affine=affine,
    )
