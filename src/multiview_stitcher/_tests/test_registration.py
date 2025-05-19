import logging

import dask.array as da
import matplotlib.pyplot
import numpy as np
import pytest
import xarray as xr
from scipy import ndimage
from skimage.exposure import rescale_intensity

from multiview_stitcher import (
    io,
    msi_utils,
    mv_graph,
    param_utils,
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
def test_pairwise_reg_against_sample_gt(pairwise_reg_func):
    example_data_path = sample_data.get_mosaic_sample_data_path()
    sims = io.read_mosaic_into_sims(example_data_path)

    sims = [
        spatial_image_utils.sim_sel_coords(sim, {"c": sim.coords["c"][0]})
        for sim in sims
    ]

    spatial_dims = spatial_image_utils.get_spatial_dims_from_sim(sims[0])

    msims = [
        msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims
    ]

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

    ### test for different dtypes and normalizations

    msimss = [msims]

    sim_extrema = [(float(sim.min()), float(sim.max())) for sim in sims]

    for out_range, dtype in zip(
        [(-1, 1), (0, 1), (-300, 0)], [np.float32, np.float32, np.int32]
    ):
        msimss += [
            [
                msi_utils.get_msim_from_sim(
                    xr.apply_ufunc(
                        rescale_intensity,
                        sim,
                        dask="allowed",
                        kwargs={
                            "in_range": sim_extrema[isim],
                            "out_range": out_range,
                        },
                        keep_attrs=True,
                    ).astype(dtype),
                    scale_factors=[],
                )
                for isim, sim in enumerate(sims)
            ]
        ]

    for msims in msimss:
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


@pytest.mark.parametrize(
    "pairwise_reg_func, translation_scale, rotation_scale, scale_scale, reg_func_kwars",
    [
        (registration.phase_correlation_registration, 10, 0, 0, {}),
        (
            registration.registration_ANTsPy,
            10,
            0.1,
            0.1,
            {"transform_types": ["Affine"]},
        ),
    ],
)
def test_pairwise_reg_against_artificial_gt(
    pairwise_reg_func,
    translation_scale,
    rotation_scale,
    scale_scale,
    reg_func_kwars,
):
    ## create artificial data
    # (better use sample_data.generate... here?)

    im = np.zeros((100, 100), dtype=float)

    im[10:40, 20:40] = 1
    im[70:80, 60:90] = 1
    im[20:40, 60:70] = 1
    im[60:80, 20:40] = 1

    im = ndimage.gaussian_filter(im, 3)

    np.random.seed(0)
    affine = param_utils.random_affine(
        ndim=2,
        translation_scale=translation_scale,
        rotation_scale=rotation_scale,
        scale_scale=scale_scale,
    )

    imt = ndimage.affine_transform(im, np.linalg.inv(affine))

    spatial_dims = ["y", "x"]

    transform_key = "metadata"
    msims = [
        msi_utils.get_msim_from_sim(
            spatial_image_utils.get_sim_from_array(
                im,
                dims=["y", "x"],
                # scale={dim: s for dim, s
                #     in zip(spatial_dims, param_utils.random_scale(ndim=2, scale=0.2))},
                # translation={dim: s for dim, s
                #     in zip(spatial_dims, param_utils.random_translation(ndim=2, scale=2))},
                transform_key=transform_key,
            ),
            scale_factors=[],
        )
        for im in [im, imt]
    ]

    msims = [
        msi_utils.multiscale_sel_coords(msim, {"c": 0, "t": 0})
        for msim in msims
    ]

    ##### test pairwise registration
    pd = registration.register_pair_of_msims_over_time(
        msims[0],
        msims[1],
        registration_binning={dim: 1 for dim in spatial_dims},
        transform_key=transform_key,
        pairwise_reg_func=pairwise_reg_func,
        pairwise_reg_func_kwargs=reg_func_kwars,
    )

    p = pd.compute()

    ## compare reg result with ground truth
    if pairwise_reg_func != registration.registration_ANTsPy:
        assert np.allclose(
            p["transform"].sel(t=0),
            affine,
            atol=0.1,
        )


def test_iterative_registration_and_transform_key_setting(monkeypatch):
    example_data_path = sample_data.get_mosaic_sample_data_path()
    sims = io.read_mosaic_into_sims(example_data_path)

    msims = [
        msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims
    ]

    registration.register(
        msims,
        transform_key=METADATA_TRANSFORM_KEY,
        new_transform_key="affine_registered",
        reg_channel_index=0,
    )

    # test return_dict=True
    registration.register(
        msims,
        transform_key="affine_registered",
        new_transform_key="affine_registered_2",
        reg_channel_index=0,
    )


@pytest.mark.parametrize(
    "plot_summary, return_dict",
    [
        (True, False),
        (True, True),
        (False, False),
        (False, True),
    ],
)
def test_plot_and_return_dict(plot_summary, return_dict, monkeypatch):
    # test plot_summary=True without plots showing up
    # https://docs.pytest.org/en/latest/how-to/monkeypatch.html
    monkeypatch.setattr(matplotlib.pyplot, "show", lambda: None)

    sims = sample_data.generate_tiled_dataset(
        ndim=2,
        overlap=2,
        N_c=1,
        N_t=2,
        tile_size=5,
        tiles_x=1,
        tiles_y=2,
        tiles_z=1,
    )

    msims = [
        msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims
    ]

    out = registration.register(
        msims,
        transform_key=METADATA_TRANSFORM_KEY,
        reg_channel_index=0,
        plot_summary=plot_summary,
        return_dict=return_dict,
    )

    if return_dict:
        assert isinstance(out, dict)
    else:
        assert isinstance(out, list)


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
            "keep_axis_aligned",
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
    caplog,
):
    caplog.set_level(logging.DEBUG)

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

    if ndim == 3:
        assert (
            "Setting n_parallel_pairwise_regs to 1 for 3D data" in caplog.text
        )
    else:
        assert (
            "Computing all pairwise registrations in parallel" in caplog.text
        )


@pytest.mark.parametrize(
    """
    groupwise_resolution_method,
    """,
    [
        "shortest_paths",
        "global_optimization",
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


@pytest.mark.parametrize(
    """
    groupwise_resolution_method,
    """,
    [
        "shortest_paths",
        "global_optimization",
        "global_optimization",
    ],
)
def test_manual_pair_registration(
    groupwise_resolution_method,
):
    # Generate a cc
    sims = sample_data.generate_tiled_dataset(
        ndim=2,
        N_t=2,
        N_c=2,
        tile_size=15,
        tiles_x=2,
        tiles_y=3,
        tiles_z=1,
        overlap=5,
    )

    msims = [
        msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims
    ]

    # choose pairs which do not represent continuous indices
    pairs = [(1, 3), (3, 2), (2, 5)]

    # Run registration
    params = registration.register(
        msims,
        reg_channel_index=0,
        transform_key=METADATA_TRANSFORM_KEY,
        pairwise_reg_func=registration.phase_correlation_registration,
        new_transform_key="affine_registered",
        groupwise_resolution_method=groupwise_resolution_method,
        pairs=pairs,
    )

    assert len(params) == 6


@pytest.mark.parametrize(
    """
    transform,
    """,
    ["translation", "rigid", "similarity", "affine"],
)
def test_global_optimization(transform):
    """
    Test the global optimization function.
    Currently only tests that the function runs without errors.
    """

    sims = sample_data.generate_tiled_dataset(
        ndim=2,
        N_t=1,
        N_c=1,
        tile_size=15,
        tiles_x=2,
        tiles_y=1,
        tiles_z=1,
        overlap=5,
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
        groupwise_resolution_method="global_optimization",
        groupwise_resolution_kwargs={"transform": transform},
    )

    if transform == "translation":
        for p in params:
            assert np.allclose(p.sel(t=0).data[:2, :2], np.eye(2))


def test_reg_channel():
    example_data_path = sample_data.get_mosaic_sample_data_path()
    sims = io.read_mosaic_into_sims(example_data_path)

    msims = [
        msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims
    ]

    registration.register(
        msims,
        transform_key=METADATA_TRANSFORM_KEY,
        reg_channel="EGFP",
    )

    registration.register(
        msims,
        transform_key=METADATA_TRANSFORM_KEY,
        reg_channel="EGFP",
        reg_channel_index=99,  # should be ignored
    )


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_prune_view_adjacency_graph():
    # create random distribution of sims
    size = 10
    N = 10

    np.random.seed(0)
    positions = np.random.random((N, 2)) * N * size / 2 - size * N / 2
    sdims = ["y", "x"]
    sims = [
        spatial_image_utils.get_sim_from_array(
            np.zeros([size] * 2, dtype=np.uint8),
            dims=sdims,
            translation=dict(zip(sdims, positions[iview])),
            transform_key=METADATA_TRANSFORM_KEY,
        )
        for iview in range(N)
    ]

    msims = [
        msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims
    ]

    # from multiview_stitcher import vis_utils
    # vis_utils.plot_positions(msims, transform_key=METADATA_TRANSFORM_KEY)

    g = mv_graph.build_view_adjacency_graph_from_msims(
        msims,
        transform_key=METADATA_TRANSFORM_KEY,
    )

    for pre_registration_pruning_method in [
        "shortest_paths_overlap_weighted",
        "otsu_threshold_on_overlap",
        "keep_axis_aligned",
    ]:
        g_reg = registration.prune_view_adjacency_graph(
            g,
            method=pre_registration_pruning_method,
        )

        assert len(g_reg.nodes) == N


def test_constant_pairwise_reg():
    sims = sample_data.generate_tiled_dataset(
        ndim=2,
        N_t=1,
        N_c=1,
        tile_size=10,
        tiles_x=2,
        tiles_y=1,
        tiles_z=1,
        overlap=5,
    )

    sims[0].data *= 0

    with pytest.warns(
        UserWarning,
        match="An overlap region between tiles/views is all zero or constant",
    ):
        registration.register(
            [
                msi_utils.get_msim_from_sim(sim, scale_factors=[])
                for sim in sims
            ],
            reg_channel_index=0,
            transform_key=METADATA_TRANSFORM_KEY,
            pairwise_reg_func=registration.phase_correlation_registration,
        )


@pytest.mark.parametrize(
    "ndim",
    [2, 3],
)
def test_overlap_tolerance(ndim):
    """
    Check that overlap_tolerance works as expected by recovering
    non overlapping images with a known shift.
    """

    overlap_x = 10
    shift_x = overlap_x

    sims = sample_data.generate_tiled_dataset(
        ndim=ndim,
        N_c=1,
        N_t=1,
        tile_size=30,
        overlap=overlap_x,
        tiles_x=2,
        tiles_y=1,
    )
    sim0, sim1 = sims

    sim1_shifted = sims[1].assign_coords(
        {"x": sims[1].coords["x"].data + shift_x}
    )

    # from multiview_stitcher import vis_utils
    # vis_utils.plot_positions([msi_utils.get_msim_from_sim(sim, scale_factors=[])
    #         for sim in
    #         [sim0, sim1_shifted]
    #         # [sim0, sim1]
    #         ],
    #     transform_key=METADATA_TRANSFORM_KEY,
    #     use_positional_colors=False,
    #     )

    params_orig = registration.register(
        [
            msi_utils.get_msim_from_sim(sim, scale_factors=[])
            for sim in [sim0, sim1]
        ],
        transform_key=METADATA_TRANSFORM_KEY,
        new_transform_key="registered_orig",
        reg_channel_index=0,
        scheduler="single-threaded",
    )

    params_shifted = registration.register(
        [
            msi_utils.get_msim_from_sim(sim, scale_factors=[])
            for sim in [sim0, sim1_shifted]
        ],
        transform_key=METADATA_TRANSFORM_KEY,
        new_transform_key="registered_shifted",
        overlap_tolerance={"x": overlap_x},
        reg_channel_index=0,
        scheduler="single-threaded",
    )

    params_diff = param_utils.translation_from_affine(
        (
            params_shifted[1]
            - params_shifted[0]
            - (params_orig[1] - params_orig[0])
        )
        .sel(t=0)
        .data
    )

    np.allclose(params_diff, [0] * (ndim - 1) + [shift_x], atol=1.5)
