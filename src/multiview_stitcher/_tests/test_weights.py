import numpy as np
import pytest

import multiview_stitcher.spatial_image_utils as si_utils
from multiview_stitcher import (
    fusion,
    io,
    param_utils,
    sample_data,
    transformation,
    weights,
)


def test_blending_weights():
    """
    Test blending weights calculation.
    """

    sims = io.read_mosaic_into_sims(sample_data.get_mosaic_sample_data_path())

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


@pytest.mark.parametrize(
    "ndim",
    [2, 3],
)
def test_blending_weight_coverage(ndim):
    """
    Check that there are no pixels that aren't contributing
    to the fused image but should be.
    """

    # create a dataset with 4 tiles, each with a different affine transform
    sims = sample_data.generate_tiled_dataset(
        ndim=ndim,
        N_c=1,
        N_t=1,
        tile_size=50,
        overlap=10,
        tiles_x=2,
        tiles_y=2,
        spacing_z=1.2,
        spacing_y=1,
        spacing_x=0.8,
    )

    np.random.seed(0)
    for _iview, sim in enumerate(sims):
        si_utils.set_sim_affine(
            sim,
            param_utils.affine_to_xaffine(
                param_utils.random_affine(
                    ndim=ndim,
                    translation_scale=0,
                    rotation_scale=0.6,
                    scale_scale=0.3,
                )
            ),
            io.METADATA_TRANSFORM_KEY,
        )

    # # visualize the positions of the tiles
    # from multiview_stitcher import vis_utils
    # msims = [msi_utils.get_msim_from_sim(sim) for sim in sims]
    # vis_utils.plot_positions(msims, transform_key=io.METADATA_TRANSFORM_KEY, use_positional_colors=False)

    fused = fusion.fuse(
        sims,
        transform_key=io.METADATA_TRANSFORM_KEY,
    ).compute()

    stack_propss = [
        si_utils.get_stack_properties_from_sim(sim) for sim in sims
    ]
    affines = [
        si_utils.get_affine_from_sim(sim, io.METADATA_TRANSFORM_KEY)
        for sim in sims
    ]
    target_bb = si_utils.get_stack_properties_from_sim(fused)

    ws = np.array(
        [
            weights.get_blending_weights(
                target_bb,
                stack_propss[iview],
                affines[iview],
            )
            for iview in range(len(sims))
        ]
    )

    simst = np.array(
        [
            transformation.transform_sim(
                sim[0, 0].astype(np.float32),
                np.linalg.inv(affines[iview]),
                target_bb,
                cval=np.nan,
            )
            for iview, sim in enumerate(sims)
        ]
    )

    # current implementation: weights might exceed transformed image region
    # and are not normalized
    ws = ws * ~np.isnan(simst)
    ws = weights.normalize_weights(ws)

    # check sum of weights are either 0 or 1
    assert np.all(
        np.isclose(np.sum(ws, 0), 1, atol=1e-5)
        | np.isclose(np.sum(ws, 0), 0, atol=1e-5)
    )

    # check that weights are strictly positive where transformed images are not nan
    # (i.e. that there are no pixels that aren't contributing to the fused image but should be)
    assert np.all(ws[~np.isnan(simst)] > 0)
