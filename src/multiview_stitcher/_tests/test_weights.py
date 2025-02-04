import multiview_stitcher.spatial_image_utils as si_utils
from multiview_stitcher import (
    io,
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

    si_utils.get_ndim_from_sim(sims[0])

    stack_propss = [
        si_utils.get_stack_properties_from_sim(sim) for sim in sims
    ]

    affines = [
        si_utils.get_affine_from_sim(
            sim, transform_key=io.METADATA_TRANSFORM_KEY
        )
        for sim in sims
    ]

    weights.get_blending_weights(
        stack_propss[0],
        stack_propss,
        affines=affines,
    )
