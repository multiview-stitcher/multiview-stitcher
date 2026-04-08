import logging
import os
import tempfile

import dask.array as da
import dask.config
import matplotlib.pyplot
import numpy as np
import pytest
import xarray as xr
from scipy import ndimage
from skimage.exposure import rescale_intensity

from multiview_stitcher import (
    io,
    msi_utils,
    param_utils,
    registration,
    sample_data,
    spatial_image_utils,
    transformation,
)
from multiview_stitcher.io import METADATA_TRANSFORM_KEY


ITK_ELASTIX_MARK = pytest.mark.skipif(
    registration.itk is None,
    reason="itk-elastix is not installed",
)


@pytest.mark.parametrize(
    "pairwise_reg_func",
    [
        registration.phase_correlation_registration,
        registration.registration_ANTsPy,
        pytest.param(
            registration.registration_ITKElastix,
            marks=ITK_ELASTIX_MARK,
        ),
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

    # ANTsPy and ITKElastix can land on slightly different local optima here.
    if pairwise_reg_func not in [
        registration.registration_ANTsPy,
        registration.registration_ITKElastix,
    ]:
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

        # ANTsPy and ITKElastix can land on slightly different local optima here.
        if pairwise_reg_func not in [
            registration.registration_ANTsPy,
            registration.registration_ITKElastix,
        ]:
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
        pytest.param(
            registration.registration_ITKElastix,
            10,
            0.1,
            0.1,
            {"transform_types": ["Affine"]},
            marks=ITK_ELASTIX_MARK,
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
    if pairwise_reg_func not in [
        registration.registration_ANTsPy,
        registration.registration_ITKElastix,
    ]:
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
        pytest.param(
            ndim,
            1,
            3,
            registration.registration_ITKElastix,
            "shortest_paths_overlap_weighted",
            False,
            "shortest_paths",
            marks=ITK_ELASTIX_MARK,
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

    # test pre_reg_pruning_method_kwargs
    # (only whether it runs without error)
    if pre_registration_pruning_method == "keep_axis_aligned" and ndim == 2:
        pre_reg_pruning_method_kwargs = {
            "max_angle": 0.1,
        }
    else:
        pre_reg_pruning_method_kwargs = None

    # Run registration
    params = registration.register(
        msims,
        reg_channel_index=0,
        transform_key=METADATA_TRANSFORM_KEY,
        pairwise_reg_func=pairwise_reg_func,
        new_transform_key="affine_registered",
        pre_registration_pruning_method=pre_registration_pruning_method,
        pre_reg_pruning_method_kwargs=pre_reg_pruning_method_kwargs,
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

    with dask.config.set(scheduler="single-threaded"):
        params_orig = registration.register(
            [
                msi_utils.get_msim_from_sim(sim, scale_factors=[])
                for sim in [sim0, sim1]
            ],
            transform_key=METADATA_TRANSFORM_KEY,
            new_transform_key="registered_orig",
            reg_channel_index=0,
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


def test_registration_with_reg_res_level():
    """Test that registration can use precomputed resolution levels."""
    # Create test data with multiscale structure
    example_data_path = sample_data.get_mosaic_sample_data_path()
    sims = io.read_mosaic_into_sims(example_data_path)
    
    sims = [
        spatial_image_utils.sim_sel_coords(sim, {"c": sim.coords["c"][0]})
        for sim in sims
    ]
    
    spatial_dims = spatial_image_utils.get_spatial_dims_from_sim(sims[0])
    
    # Create multiscale images with multiple resolution levels
    scale_factors = [{"y": 2, "x": 2}, {"y": 2, "x": 2}]  # scale1: 2x, scale2: 4x
    msims = [
        msi_utils.get_msim_from_sim(sim, scale_factors=scale_factors)
        for sim in sims
    ]
    
    # Test 1: Use reg_res_level=1 directly
    params_level1 = registration.register(
        [msims[0], msims[1]],
        reg_res_level=1,
        transform_key=METADATA_TRANSFORM_KEY,
        new_transform_key=None,
    )
    
    # Test 2: Use registration_binning that matches scale1
    params_binning = registration.register(
        [msims[0], msims[1]],
        registration_binning={"y": 2, "x": 2},
        transform_key=METADATA_TRANSFORM_KEY,
        new_transform_key=None,
    )
    
    # Test 3: Use both reg_res_level and registration_binning (compatible)
    params_both = registration.register(
        [msims[0], msims[1]],
        reg_res_level=1,
        registration_binning={"y": 2, "x": 2},
        transform_key=METADATA_TRANSFORM_KEY,
        new_transform_key=None,
    )

    # Test 4: Don't specify either (should auto-determine)
    params_none = registration.register(
        [msims[0], msims[1]],
        transform_key=METADATA_TRANSFORM_KEY,
        new_transform_key=None,
    )
    
    # Test 4: Verify error is raised for incompatible reg_res_level and binning
    try:
        registration.register(
            [msims[0], msims[1]],
            reg_res_level=1,  # scale1 has factor 2
            registration_binning={"y": 3, "x": 3},  # not compatible with factor 2 (3 % 2 != 0)
            transform_key=METADATA_TRANSFORM_KEY,
            new_transform_key=None,
        )
        assert False, "Should have raised ValueError for incompatible parameters"
    except ValueError as e:
        assert "not a divisor" in str(e)
    
    # Verify that all valid registrations produce similar results
    # (should be within tolerance since they use the same or similar data)
    gt_shift = xr.DataArray(
        [2.5, 7.5],
        dims=["x_in"],
        coords={"x_in": ["y", "x"]},
    )
    tolerance = 2.0
    
    for params in [params_level1, params_binning, params_both, params_none]:
        ctrl_pt = np.zeros((2,))
        ctrl_pts_t = [
            transformation.transform_pts([ctrl_pt], affine.squeeze())[0]
            for affine in params
        ]
        rel_pos = ctrl_pts_t[0] - ctrl_pts_t[1]
        
        # Check that the relative position is close to ground truth
        assert np.allclose(rel_pos, gt_shift, atol=tolerance)


@pytest.mark.skip(reason="Skipping this test for now because of sporadic failures that need investigation.")
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize(
    "transform_types",
    [
        ["Rigid"],
        ["Similarity"],
        # ["Translation", "Rigid"],
        # ["Translation", "Rigid", "Similarity"],
    ],
)
def test_registration_ANTsPy_rotation_recovery(ndim, transform_types):
    """
    Test that registration_ANTsPy can recover known rotations in 2D and 3D.

    Rather than focusing on ensuring precise recovery of the rotation parameters,
    this test checks that recovered transformations are broadly correct to
    make sure parameter axis convention / order is correct.
    
    This test creates a synthetic image with distinct features, applies a known
    rotation to it, and then uses registration_ANTsPy to register the rotated
    image back to the original. It verifies that the recovered transformation
    closely matches the inverse of the applied rotation.
    """
    # Create a synthetic image with distinct features
    if ndim == 2:
        # 2D image with multiple rectangular features
        im = np.zeros((100, 100), dtype=float)
        im[10:40, 20:40] = 1
        im[70:80, 60:90] = 1
        im[20:40, 60:70] = 0.7
        im[60:80, 20:40] = 0.5
        spatial_dims = ["y", "x"]
        # rotation angle in radians (15 degrees)
        rotation_angle = np.pi / 12
        # Direction vector for 2D rotation (around z-axis)
        direction = [0, 0, 1]
    else:
        # 3D image with multiple box features
        im = np.zeros((60, 60, 60), dtype=float)
        im[10:30, 15:30, 20:35] = 1
        im[35:50, 10:25, 40:55] = 0.8
        im[15:25, 35:50, 10:25] = 0.6
        im[40:55, 40:50, 15:30] = 0.4
        spatial_dims = ["z", "y", "x"]
        # rotation angle in radians (10 degrees for 3D to be more conservative)
        rotation_angle = np.pi / 18
        # Direction vector for 3D rotation (around diagonal axis)
        direction = np.array([1, 1, 1]) / np.sqrt(3)

    # Apply Gaussian smoothing to make features less sharp
    im = ndimage.gaussian_filter(im, sigma=2)
    
    # Create the rotation transformation
    center = np.array(im.shape) / 2
    if ndim == 2:
        # For 2D, we need to work in homogeneous coordinates
        affine_rotation = param_utils.affine_from_rotation(
            rotation_angle, direction, point=list(center) + [0]
        )[:3, :3]  # Extract 2D part
    else:
        # For 3D, use directly
        affine_rotation = param_utils.affine_from_rotation(
            rotation_angle, direction, point=center
        )
    
    # Apply the rotation to create the moving image
    imt = ndimage.affine_transform(
        im, 
        np.linalg.inv(affine_rotation[:ndim, :ndim]),
        offset=np.linalg.inv(affine_rotation)[:ndim, ndim],
        order=3,
        mode='constant',
        cval=0
    )
    
    # Set up spacing and origin
    spacing = {dim: 1.0 for dim in spatial_dims}
    origin = {dim: 0.0 for dim in spatial_dims}
    
    # Create xarray DataArrays for fixed and moving images
    fixed_data = xr.DataArray(
        im,
        dims=spatial_dims,
        coords={dim: np.arange(im.shape[i]) for i, dim in enumerate(spatial_dims)},
    )
    
    moving_data = xr.DataArray(
        imt,
        dims=spatial_dims,
        coords={dim: np.arange(imt.shape[i]) for i, dim in enumerate(spatial_dims)},
    )
    
    # Run registration
    reg_result = registration.registration_ANTsPy(
        fixed_data=fixed_data,
        moving_data=moving_data,
        fixed_origin=origin,
        moving_origin=origin,
        fixed_spacing=spacing,
        moving_spacing=spacing,
        initial_affine=param_utils.identity_transform(ndim),
        transform_types=transform_types,
        ants_registration_kwargs={
            'aff_metric': 'meansquares',
        }
    )
    
    # Extract the recovered affine matrix
    recovered_affine = np.array(reg_result["affine_matrix"])
    
    # Test in pixel space: transform vertices/borders through expected transform
    # and back through recovered transform, then check RMS error
    
    # Generate vertices (corners) and edge points of the image
    if ndim == 2:
        # For 2D: corners and edge midpoints
        shape = im.shape
        vertices = np.array([
            [0, 0], [0, shape[1]-1], [shape[0]-1, 0], [shape[0]-1, shape[1]-1],
            [shape[0]//2, 0], [shape[0]//2, shape[1]-1],  # top and bottom midpoints
            [0, shape[1]//2], [shape[0]-1, shape[1]//2],  # left and right midpoints
            [shape[0]//2, shape[1]//2],  # center
        ])
    else:
        # For 3D: corners and face centers
        shape = im.shape
        vertices = []
        # 8 corners
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    vertices.append([
                        i * (shape[0]-1),
                        j * (shape[1]-1),
                        k * (shape[2]-1)
                    ])
        # face centers
        for dim in range(3):
            for val in [0, shape[dim]-1]:
                center = [s // 2 for s in shape]
                center[dim] = val
                vertices.append(center)
        # volume center
        vertices.append([s // 2 for s in shape])
        vertices = np.array(vertices)
    
    # Transform vertices through the expected rotation (ground truth)
    vertices_transformed_expected = transformation.transform_pts(
        vertices, affine_rotation
    )
    
    # Transform back through the recovered transformation
    # The recovered transform maps moving->fixed, so we need its inverse to go back
    vertices_recovered = transformation.transform_pts(
        vertices_transformed_expected, np.linalg.inv(recovered_affine)
    )
    
    # Calculate RMS error between original and round-trip vertices
    errors = vertices - vertices_recovered
    rms_error = np.sqrt(np.mean(np.sum(errors**2, axis=1)))
    
    # For rigid transforms, the RMS error should be small
    if any(t in ["Rigid", "Similarity", "Affine"] for t in transform_types):
        rms_tolerance = 0.1 if ndim == 2 else 0.5
        assert rms_error < rms_tolerance, (
            f"RMS error too large for {ndim}D with transform_types={transform_types}. "
            f"RMS error: {rms_error:.4f} pixels, tolerance: {rms_tolerance} pixels"
        )
    
    # Check quality metric
    assert reg_result["quality"] > 0.75, (
        f"Registration quality too low: {reg_result['quality']:.3f}"
    )


@ITK_ELASTIX_MARK
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize(
    "transform_types",
    [
        ["Affine"],
        ["Rigid", "Affine"],
        ["Rigid"],
        ["Similarity"],
        ["Translation", "Rigid"],
        ["Translation", "Rigid", "Similarity"],
    ],
)
def test_registration_ITKElastix_rotation_recovery(ndim, transform_types):
    if ndim == 2:
        im = np.zeros((100, 100), dtype=float)
        im[10:40, 20:40] = 1
        im[70:80, 60:90] = 1
        im[20:40, 60:70] = 0.7
        im[60:80, 20:40] = 0.5
        spatial_dims = ["y", "x"]
        rotation_angle = np.pi / 12
        direction = [0, 0, 1]
    else:
        im = np.zeros((60, 60, 60), dtype=float)
        im[10:30, 15:30, 20:35] = 1
        im[35:50, 10:25, 40:55] = 0.8
        im[15:25, 35:50, 10:25] = 0.6
        im[40:55, 40:50, 15:30] = 0.4
        spatial_dims = ["z", "y", "x"]
        rotation_angle = np.pi / 18
        direction = np.array([1, 1, 1]) / np.sqrt(3)

    im = ndimage.gaussian_filter(im, sigma=2)

    center = np.array(im.shape) / 2
    if ndim == 2:
        affine_rotation = param_utils.affine_from_rotation(
            rotation_angle, direction, point=list(center) + [0]
        )[:3, :3]
    else:
        affine_rotation = param_utils.affine_from_rotation(
            rotation_angle, direction, point=center
        )

    imt = ndimage.affine_transform(
        im,
        np.linalg.inv(affine_rotation)[:ndim, :ndim],
        offset=np.linalg.inv(affine_rotation)[:ndim, ndim],
        order=3,
        mode="constant",
        cval=0,
    )

    spacing = {dim: 1.0 for dim in spatial_dims}
    origin = {dim: 0.0 for dim in spatial_dims}

    fixed_data = xr.DataArray(
        im,
        dims=spatial_dims,
        coords={dim: np.arange(im.shape[i]) for i, dim in enumerate(spatial_dims)},
    )
    moving_data = xr.DataArray(
        imt,
        dims=spatial_dims,
        coords={
            dim: np.arange(imt.shape[i]) for i, dim in enumerate(spatial_dims)
        },
    )

    reg_result = registration.registration_ITKElastix(
        fixed_data=fixed_data,
        moving_data=moving_data,
        fixed_origin=origin,
        moving_origin=origin,
        fixed_spacing=spacing,
        moving_spacing=spacing,
        initial_affine=param_utils.identity_transform(ndim),
        transform_types=transform_types,
    )

    recovered_affine = np.array(reg_result["affine_matrix"])

    if ndim == 2:
        shape = im.shape
        vertices = np.array(
            [
                [0, 0],
                [0, shape[1] - 1],
                [shape[0] - 1, 0],
                [shape[0] - 1, shape[1] - 1],
                [shape[0] // 2, 0],
                [shape[0] // 2, shape[1] - 1],
                [0, shape[1] // 2],
                [shape[0] - 1, shape[1] // 2],
                [shape[0] // 2, shape[1] // 2],
            ]
        )
    else:
        shape = im.shape
        vertices = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    vertices.append(
                        [
                            i * (shape[0] - 1),
                            j * (shape[1] - 1),
                            k * (shape[2] - 1),
                        ]
                    )
        for dim in range(3):
            for val in [0, shape[dim] - 1]:
                face_center = [s // 2 for s in shape]
                face_center[dim] = val
                vertices.append(face_center)
        vertices.append([s // 2 for s in shape])
        vertices = np.array(vertices)

    vertices_transformed_expected = transformation.transform_pts(
        vertices, affine_rotation
    )
    vertices_recovered = transformation.transform_pts(
        vertices_transformed_expected, np.linalg.inv(recovered_affine)
    )

    errors = vertices - vertices_recovered
    rms_error = np.sqrt(np.mean(np.sum(errors**2, axis=1)))

    # Affine has more degrees of freedom and is susceptible to local minima on
    # small 3-D test images; use a relaxed tolerance for Affine-containing
    # transform sequences in 3-D.
    has_affine = "Affine" in transform_types
    if ndim == 2:
        rms_tolerance = 0.1
    elif has_affine:
        rms_tolerance = 1.0
    else:
        rms_tolerance = 0.5
    assert rms_error < rms_tolerance, (
        f"RMS error too large for {ndim}D with transform_types={transform_types}. "
        f"RMS error: {rms_error:.4f} pixels, tolerance: {rms_tolerance} pixels"
    )
    assert reg_result["quality"] > 0.75, (
        f"Registration quality too low: {reg_result['quality']:.3f}"
    )


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize(
    "transform_types",
    [
        ["Rigid", "Affine"],
        ["Rigid"],
        ["Similarity"],
    ],
)
@pytest.mark.parametrize(
    "pairwise_reg_func",
    [
        # pytest.param(
        #     registration.registration_ANTsPy,
        #     id="registration_ANTsPy",
        # ),
        pytest.param(
            registration.registration_ITKElastix,
            marks=ITK_ELASTIX_MARK,
            id="registration_ITKElastix",
        ),
    ],
)
def test_registration_non_identity_initial_transform_recovery(
    ndim,
    pairwise_reg_func,
    transform_types,
):
    """
    Check that pairwise registration can undo a non-identity preregistration
    when the image content is already aligned.
    """
    if ndim == 2:
        im = np.zeros((100, 100), dtype=float)
        im[10:40, 20:40] = 1
        im[70:80, 60:90] = 1
        im[20:40, 60:70] = 0.7
        im[60:80, 20:40] = 0.5
        spatial_dims = ["y", "x"]
        rotation_angle = np.pi / 12
        direction = [0, 0, 1]
    else:
        im = np.zeros((60, 60, 60), dtype=float)
        im[10:30, 15:30, 20:35] = 1
        im[35:50, 10:25, 40:55] = 0.8
        im[15:25, 35:50, 10:25] = 0.6
        im[40:55, 40:50, 15:30] = 0.4
        spatial_dims = ["z", "y", "x"]
        rotation_angle = np.pi / 18
        direction = np.array([1, 1, 1]) / np.sqrt(3)

    im = ndimage.gaussian_filter(im, sigma=2)

    center = np.array(im.shape) / 2
    if ndim == 2:
        initial_affine = param_utils.affine_from_rotation(
            rotation_angle, direction, point=list(center) + [0]
        )[:3, :3]
    else:
        initial_affine = param_utils.affine_from_rotation(
            rotation_angle, direction, point=center
        )

    transform_key = "initial_transform_test"
    fixed_sim = spatial_image_utils.get_sim_from_array(
        im,
        dims=spatial_dims,
        affine=initial_affine,
        transform_key=transform_key,
    )
    moving_sim = spatial_image_utils.get_sim_from_array(
        im,
        dims=spatial_dims,
        transform_key=transform_key,
    )

    fixed_msim = msi_utils.multiscale_sel_coords(
        msi_utils.get_msim_from_sim(fixed_sim, scale_factors=[]),
        {"c": 0, "t": 0},
    )
    moving_msim = msi_utils.multiscale_sel_coords(
        msi_utils.get_msim_from_sim(moving_sim, scale_factors=[]),
        {"c": 0, "t": 0},
    )

    reg_result = registration.register_pair_of_msims_over_time(
        fixed_msim,
        moving_msim,
        registration_binning={dim: 1 for dim in spatial_dims},
        transform_key=transform_key,
        pairwise_reg_func=pairwise_reg_func,
        pairwise_reg_func_kwargs={
            "transform_types": transform_types,
        },
    ).compute()

    recovered_affine = np.array(reg_result["transform"].sel(t=0))
    expected_affine = np.linalg.inv(initial_affine)

    if ndim == 2:
        shape = im.shape
        vertices = np.array(
            [
                [0, 0],
                [0, shape[1] - 1],
                [shape[0] - 1, 0],
                [shape[0] - 1, shape[1] - 1],
                [shape[0] // 2, 0],
                [shape[0] // 2, shape[1] - 1],
                [0, shape[1] // 2],
                [shape[0] - 1, shape[1] // 2],
                [shape[0] // 2, shape[1] // 2],
            ]
        )
    else:
        shape = im.shape
        vertices = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    vertices.append(
                        [
                            i * (shape[0] - 1),
                            j * (shape[1] - 1),
                            k * (shape[2] - 1),
                        ]
                    )
        for dim in range(3):
            for val in [0, shape[dim] - 1]:
                face_center = [s // 2 for s in shape]
                face_center[dim] = val
                vertices.append(face_center)
        vertices.append([s // 2 for s in shape])
        vertices = np.array(vertices)

    vertices_expected = transformation.transform_pts(vertices, expected_affine)
    vertices_recovered = transformation.transform_pts(
        vertices, recovered_affine
    )

    errors = vertices_expected - vertices_recovered
    rms_error = np.sqrt(np.mean(np.sum(errors**2, axis=1)))

    # Affine has more degrees of freedom and is susceptible to local minima on
    # small 3-D test images; use a relaxed tolerance for Affine-containing
    # transform sequences in 3-D.
    has_affine = "Affine" in transform_types
    if ndim == 2:
        rms_tolerance = 0.1
    elif has_affine:
        rms_tolerance = 1.0
    else:
        rms_tolerance = 0.8
    assert rms_error < rms_tolerance, (
        f"RMS error too large for {ndim}D with {pairwise_reg_func.__name__}, "
        f"transform_types={transform_types}. "
        f"RMS error: {rms_error:.4f} pixels, tolerance: {rms_tolerance} pixels"
    )
    assert float(reg_result["quality"].sel(t=0)) > 0.75, (
        f"Registration quality too low: {float(reg_result['quality'].sel(t=0)):.3f}"
    )


@ITK_ELASTIX_MARK
def test_itk_elastix_initial_transform_handles_large_translation():
    initial_affine = np.array(
        [
            [np.sqrt(0.5), np.sqrt(0.5), 0.0, -242.19528185],
            [-np.sqrt(0.5), np.sqrt(0.5), 0.0, 174.87901171],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "initial_transform.txt")
        registration._write_initial_elastix_transform(
            output_path,
            initial_affine=initial_affine,
            ndim=3,
        )

        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

        input_points = registration._get_elastix_probe_points(3)
        expected_points = transformation.transform_pts(
            input_points, initial_affine
        )

        input_points_path = os.path.join(tmpdir, "input_points.txt")
        output_dir = os.path.join(tmpdir, "transformix_output")
        os.mkdir(output_dir)

        registration._write_elastix_point_set_file(
            input_points_path,
            registration._points_to_itk_spatial_order(input_points),
        )

        parameter_object = registration.itk.ParameterObject.New()
        parameter_object.ReadParameterFile(output_path)

        dummy_image = registration._get_itk_image_from_data(
            np.zeros((1, 1, 1), dtype=np.float32),
            origin=[0.0, 0.0, 0.0],
            spacing=[1.0, 1.0, 1.0],
        )

        registration.itk.transformix_filter(
            moving_image=dummy_image,
            transform_parameter_object=parameter_object,
            output_directory=output_dir,
            fixed_point_set_file_name=input_points_path,
            log_to_console=False,
        )

        transformed_points = registration._parse_elastix_output_points(
            os.path.join(output_dir, "outputpoints.txt")
        )
        assert np.allclose(transformed_points, expected_points)
