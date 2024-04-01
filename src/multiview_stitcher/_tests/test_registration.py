import dask.array as da
import numpy as np
import pytest
import xarray as xr

from multiview_stitcher import (
    io,
    msi_utils,
    registration,
    sample_data,
    spatial_image_utils,
    transformation,
)
from multiview_stitcher.io import METADATA_TRANSFORM_KEY


@pytest.mark.parametrize(
    "pairwise_reg_func",
    [
        registration.phase_correlation_registration,
        registration.registration_ANTsPy,
    ],
)
def test_reg_against_gt(pairwise_reg_func):
    example_data_path = sample_data.get_mosaic_sample_data_path()
    sims = io.read_mosaic_image_into_list_of_spatial_xarrays(example_data_path)

    sims = [
        spatial_image_utils.sim_sel_coords(sim, {"c": sim.coords["c"][0]})
        for sim in sims
    ]

    msims = [
        msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims
    ]

    spatial_dims = spatial_image_utils.get_spatial_dims_from_sim(sims[0])

    ##### test pairwise registration
    pd = registration.register_pair_of_msims_over_time(
        msims[0],
        msims[1],
        registration_binning={dim: 1 for dim in spatial_dims},
        transform_key=METADATA_TRANSFORM_KEY,
        pairwise_reg_func=pairwise_reg_func,
    )

    p = pd.compute()

    # assert matrix
    assert np.allclose(
        # p["transform"].sel(t=0, x_in=[0, 1], x_out=[0, 1]),
        p["transform"].sel(t=0, x_in=["x", "y"], x_out=["x", "y"]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        atol=0.05,
    )

    gt_shift = xr.DataArray(
        [2.5, 7.5],
        dims=["x_in"],
        coords={"x_in": ["y", "x"]},
    )
    tolerance = 1.5

    # somehow antspy sporadically yields different results in ~1/10 times
    if pairwise_reg_func != registration.registration_ANTsPy:
        # assert offset
        assert np.allclose(
            p["transform"].sel(t=0, x_in=["y", "x"], x_out="1") - gt_shift,
            np.zeros((2,)),
            atol=tolerance,
        )

    ##### test groupwise registration
    p = registration.register(
        [msims[0], msims[1]],
        registration_binning={dim: 1 for dim in spatial_dims},
        transform_key=METADATA_TRANSFORM_KEY,
        pairwise_reg_func=pairwise_reg_func,
    )

    # for groupwise registration, check relative position of a control point
    ctrl_pt = np.zeros((2,))
    ctrl_pts_t = [
        transformation.transform_pts([ctrl_pt], affine.squeeze())[0]
        for affine in p
    ]
    rel_pos = ctrl_pts_t[0] - ctrl_pts_t[1]

    # somehow antspy sporadically yields different results in ~1/10 times
    if pairwise_reg_func != registration.registration_ANTsPy:
        # assert offset
        assert np.allclose(
            rel_pos,
            # np.array([2.5, 7.5]),
            gt_shift,
            atol=1.5,
        )


def test_iterative_registration_and_transform_key_setting():
    example_data_path = sample_data.get_mosaic_sample_data_path()
    sims = io.read_mosaic_image_into_list_of_spatial_xarrays(example_data_path)

    msims = [
        msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims
    ]

    registration.register(
        msims,
        transform_key=METADATA_TRANSFORM_KEY,
        new_transform_key="affine_registered",
        reg_channel_index=0,
    )

    registration.register(
        msims,
        transform_key="affine_registered",
        new_transform_key="affine_registered_2",
        reg_channel_index=0,
    )


@pytest.mark.parametrize("ndim", [2, 3])
def test_register_with_single_pixel_overlap(ndim):
    sims = sample_data.generate_tiled_dataset(
        ndim=ndim,
        overlap=1,
        N_c=1,
        N_t=2,
        tile_size=10,
        tiles_x=1,
        tiles_y=2,
        tiles_z=1,
        spacing_x=1,
        spacing_y=1,
        spacing_z=1,
    )

    sims = [
        spatial_image_utils.sim_sel_coords(sim, {"c": sim.coords["c"][0]})
        for sim in sims
    ]

    msims = [
        msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims
    ]

    registration.register_pair_of_msims_over_time(
        msims[0],
        msims[1],
        transform_key=METADATA_TRANSFORM_KEY,
    )


def test_get_optimal_registration_binning():
    ndim = 3
    sims = [
        xr.DataArray(
            da.empty([1000] * ndim), dims=spatial_image_utils.SPATIAL_DIMS
        )
        for _ in range(2)
    ]

    reg_binning = registration.get_optimal_registration_binning(*tuple(sims))

    assert min(reg_binning.values()) > 1
    assert max(reg_binning.values()) < 4


@pytest.mark.parametrize(
    """
    ndim, N_c, N_t,
    pairwise_reg_func,
    pre_registration_pruning_method,
    post_registration_do_quality_filter,
    groupwise_resolution_method,
    """,
    [
        (
            ndim,
            1,
            3,
            registration.registration_ANTsPy,
            "shortest_paths_overlap_weighted",
            False,
            "shortest_paths",
        )
        for ndim in [2, 3]
    ]
    + [
        (
            ndim,
            1,
            3,
            registration.phase_correlation_registration,
            pre_reg_pm,
            True,
            groupwise_resolution_method,
        )
        for pre_reg_pm in [
            "shortest_paths_overlap_weighted",
            "otsu_threshold_on_overlap",
            None,
        ]
        for ndim in [2, 3]
        for groupwise_resolution_method in [
            "shortest_paths",
            "global_optimization",
        ]
    ],
)
def test_register(
    ndim,
    N_c,
    N_t,
    pairwise_reg_func,
    pre_registration_pruning_method,
    post_registration_do_quality_filter,
    groupwise_resolution_method,
):
    sims = sample_data.generate_tiled_dataset(
        ndim=ndim,
        N_t=N_t,
        N_c=N_c,
        tile_size=15,
        tiles_x=2,
        tiles_y=1,
        tiles_z=2,
        zoom=10,
    )

    msims = [
        msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims
    ]

    # Run registration
    params = registration.register(
        msims,
        reg_channel_index=0,
        transform_key=METADATA_TRANSFORM_KEY,
        pairwise_reg_func=pairwise_reg_func,
        new_transform_key="affine_registered",
        pre_registration_pruning_method=pre_registration_pruning_method,
        post_registration_do_quality_filter=post_registration_do_quality_filter,
        post_registration_quality_threshold=-1,
    )

    assert len(params) == 2 ** (ndim - 1)


@pytest.mark.parametrize(
    """
    groupwise_resolution_method,
    """,
    [
        "shortest_paths",
        "global_optimization",
    ],
)
def test_cc_registration(
    groupwise_resolution_method,
):
    # Generate a cc
    sims = sample_data.generate_tiled_dataset(
        ndim=2,
        N_t=2,
        N_c=2,
        tile_size=15,
        tiles_x=3,
        tiles_y=1,
        tiles_z=1,
        overlap=5,
    )

    # remove last tile from cc
    sims[2] = sims[2].assign_coords(
        {"y": sims[2].coords["y"] + max(sims[2].coords["y"]) + 1}
    )

    msims = [
        msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims
    ]

    # Run registration
    params = registration.register(
        msims,
        reg_channel_index=0,
        transform_key=METADATA_TRANSFORM_KEY,
        pairwise_reg_func=registration.phase_correlation_registration,
        new_transform_key="affine_registered",
        groupwise_resolution_method=groupwise_resolution_method,
    )

    assert len(params) == 3
