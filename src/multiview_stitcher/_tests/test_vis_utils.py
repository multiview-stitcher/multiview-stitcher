import os
import tempfile

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib import pyplot as plt

import multiview_stitcher.spatial_image_utils as si_utils
from multiview_stitcher import (
    io,
    msi_utils,
    ngff_utils,
    param_utils,
    sample_data,
    vis_utils,
)
from multiview_stitcher.io import METADATA_TRANSFORM_KEY

@pytest.mark.parametrize(
    "ndim, N_t",
    [(2, 1), (2, 2), (3, 1), (3, 2)],
)
def test_plot_positions(ndim, N_t, monkeypatch):
    sims = sample_data.generate_tiled_dataset(
        ndim=ndim,
        N_c=2,
        N_t=N_t,
        tile_size=5,
        tiles_x=2,
        tiles_y=2,
        tiles_z=2,
    )

    msims = [
        msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims
    ]

    monkeypatch.setattr(plt, "show", lambda: None)

    indexed_sims1 = set(sims[0].coords.xindexes.dims)

    fig, ax = vis_utils.plot_positions(
        sims, transform_key=METADATA_TRANSFORM_KEY
    )

    fig, ax = vis_utils.plot_positions(
        msims, transform_key=METADATA_TRANSFORM_KEY
    )

    assert len(ax.collections) == len(msims)

    indexed_sims2 = set(sims[0].coords.xindexes.dims)

    assert indexed_sims1 == indexed_sims2


def test_plot_positions_single_coord(monkeypatch):
    sim = si_utils.get_sim_from_array(
        np.random.randint(0, 255, (1, 100, 100)),
        dims=["z", "y", "x"],
    )

    monkeypatch.setattr(plt, "show", lambda: None)

    vis_utils.plot_positions(
        [msi_utils.get_msim_from_sim(sim, scale_factors=[])],
        transform_key=si_utils.DEFAULT_TRANSFORM_KEY,
        use_positional_colors=False,
    )

    vis_utils.plot_positions(
        [msi_utils.get_msim_from_sim(sim, scale_factors=[])],
        transform_key=si_utils.DEFAULT_TRANSFORM_KEY,
        use_positional_colors=False,
        spacing={"x": 1, "y": 1, "z": 1},
    )


def test_plot_positions_point_sets(monkeypatch):
    affine = param_utils.affine_from_translation([10.0, 20.0])
    sim = si_utils.get_sim_from_array(
        np.zeros((10, 10)),
        dims=["y", "x"],
        affine=affine,
    )
    si_utils.set_point_set(
        sim,
        np.array([[1.0, 2.0], [3.0, 4.0]]),
    )

    monkeypatch.setattr(plt, "show", lambda: None)

    point_positions = vis_utils._get_point_set_positions_for_plot(
        sim,
        points_key="beads",
        transform_key=si_utils.DEFAULT_TRANSFORM_KEY,
        sdims=["y", "x"],
    )
    assert np.allclose(
        point_positions,
        np.array([[0.0, 22.0, 11.0], [0.0, 24.0, 13.0]]),
    )

    fig, ax = vis_utils.plot_positions(
        [sim],
        transform_key=si_utils.DEFAULT_TRANSFORM_KEY,
        points_key="beads",
    )
    assert len(ax.collections) == 2
    plt.close(fig)


def test_imshow_msim_with_points(monkeypatch):
    data = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6)
    sim = si_utils.get_sim_from_array(
        data,
        dims=["z", "y", "x"],
        scale={"z": 2.0, "y": 3.0, "x": 4.0},
        translation={"z": 10.0, "y": 20.0, "x": 30.0},
    )
    msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
    msi_utils.set_point_set(
        msim,
        np.array(
            [
                [10.0, 20.0, 30.0],
                [14.0, 23.0, 34.0],
                [16.0, 24.5, 34.0],
            ]
        ),
        points_key="beads",
    )

    monkeypatch.setattr(plt, "show", lambda: None)

    fig, ax, sliders = vis_utils.imshow(
        msim,
        points_key="beads",
        points_tolerance=1,
        figure_kwargs={"figsize": (4, 3)},
        imshow_kwargs={"vmin": 0, "vmax": 100},
        scatter_kwargs={"s": 12, "edgecolor": "cyan"},
    )

    assert list(sliders.keys()) == ["z"]
    np.testing.assert_allclose(fig.get_size_inches(), np.array([4.0, 3.0]))
    assert ax.images[0].get_clim() == (0, 100)
    np.testing.assert_allclose(ax.collections[0].get_sizes(), np.array([12]))
    np.testing.assert_allclose(
        np.asarray(ax.images[0].get_extent(), dtype=float),
        np.array([28.0, 52.0, 33.5, 18.5]),
    )
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"
    np.testing.assert_allclose(
        ax.collections[0].get_offsets(),
        np.array([[30.0, 20.0]]),
    )

    sliders["z"].set_val(2)
    np.testing.assert_allclose(
        ax.collections[0].get_offsets(),
        np.array([[34.0, 23.0], [34.0, 24.5]]),
    )
    plt.close(fig)


def test_imshow_without_points(monkeypatch):
    data = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6)
    sim = si_utils.get_sim_from_array(
        data,
        dims=["z", "y", "x"],
        scale={"z": 2.0, "y": 3.0, "x": 4.0},
        translation={"z": 10.0, "y": 20.0, "x": 30.0},
    )
    msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
    msi_utils.set_point_set(
        msim,
        np.array(
            [
                [10.0, 20.0, 30.0],
                [14.0, 23.0, 34.0],
            ]
        ),
        points_key="beads",
    )

    monkeypatch.setattr(plt, "show", lambda: None)

    fig, ax, sliders = vis_utils.imshow(
        msim,
        points_key=None,
    )

    assert list(sliders.keys()) == ["z"]
    assert len(ax.collections) == 0
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"
    assert "points=" not in ax.get_title()
    plt.close(fig)


@pytest.mark.parametrize(
    "project_dim, expected_image, expected_offsets, expected_labels, expected_extent",
    [
        (
            "z",
            lambda data: data.max(axis=0),
            np.array([[30.0, 20.0], [34.0, 23.0], [34.0, 24.5]]),
            ("x", "y"),
            np.array([28.0, 52.0, 33.5, 18.5]),
        ),
        (
            "y",
            lambda data: data.max(axis=1),
            np.array([[30.0, 10.0], [34.0, 14.0], [34.0, 16.0]]),
            ("x", "z"),
            np.array([28.0, 52.0, 17.0, 9.0]),
        ),
        (
            "x",
            lambda data: data.max(axis=2).T,
            np.array([[10.0, 20.0], [14.0, 23.0], [16.0, 24.5]]),
            ("z", "y"),
            np.array([9.0, 17.0, 33.5, 18.5]),
        ),
    ],
)
def test_imshow_projection(
    monkeypatch,
    project_dim,
    expected_image,
    expected_offsets,
    expected_labels,
    expected_extent,
):
    data = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6)
    sim = si_utils.get_sim_from_array(
        data,
        dims=["z", "y", "x"],
        scale={"z": 2.0, "y": 3.0, "x": 4.0},
        translation={"z": 10.0, "y": 20.0, "x": 30.0},
    )
    msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
    msi_utils.set_point_set(
        msim,
        np.array(
            [
                [10.0, 20.0, 30.0],
                [14.0, 23.0, 34.0],
                [16.0, 24.5, 34.0],
            ]
        ),
        points_key="beads",
    )

    monkeypatch.setattr(plt, "show", lambda: None)

    fig, ax, sliders = vis_utils.imshow(
        msim,
        points_key="beads",
        project_dim=project_dim,
    )

    assert list(sliders.keys()) == []
    np.testing.assert_allclose(ax.images[0].get_array(), expected_image(data))
    np.testing.assert_allclose(ax.collections[0].get_offsets(), expected_offsets)
    np.testing.assert_allclose(
        np.asarray(ax.images[0].get_extent(), dtype=float),
        expected_extent,
    )
    assert ax.get_xlabel() == expected_labels[0]
    assert ax.get_ylabel() == expected_labels[1]
    assert f"project={project_dim}" in ax.get_title()
    plt.close(fig)


def test_imshow_points_key_and_resolution_behavior(monkeypatch):
    sim = si_utils.get_sim_from_array(
        np.zeros((5, 6)),
        dims=["y", "x"],
    )
    msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])

    monkeypatch.setattr(plt, "show", lambda: None)

    fig, ax, sliders = vis_utils.imshow(msim, points_key=None)
    assert list(sliders.keys()) == []
    assert len(ax.collections) == 0
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"
    plt.close(fig)

    msi_utils.set_point_set(msim, np.array([[1.0, 2.0]]), points_key="beads")
    msi_utils.set_point_set(msim, np.array([[3.0, 4.0]]), points_key="spots")

    fig, ax, sliders = vis_utils.imshow(msim, points_key=None)
    assert list(sliders.keys()) == []
    assert len(ax.collections) == 0
    plt.close(fig)

    fig, ax, sliders = vis_utils.imshow(
        msim,
        points_key="beads",
    )
    assert list(sliders.keys()) == []
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"
    plt.close(fig)

    with pytest.raises(ValueError, match="Point set 'missing' not found"):
        vis_utils.imshow(msim, points_key="missing")

    with pytest.raises(ValueError, match="horizontal_dim must be one of"):
        vis_utils.imshow(
            msim,
            points_key="beads",
            horizontal_dim="z",
        )

    with pytest.raises(ValueError, match="requires two displayed spatial dimensions"):
        vis_utils.imshow(
            msim,
            points_key="beads",
            project_dim="y",
        )

    with pytest.raises(ValueError, match="resolution_level is only supported"):
        vis_utils.imshow(sim, resolution_level=1)


def test_imshow_accepts_sim_and_custom_axes(monkeypatch):
    data = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6)
    sim = si_utils.get_sim_from_array(
        data,
        dims=["z", "y", "x"],
        scale={"z": 2.0, "y": 3.0, "x": 4.0},
        translation={"z": 10.0, "y": 20.0, "x": 30.0},
    )
    si_utils.set_point_set(
        sim,
        np.array(
            [
                [10.0, 20.0, 30.0],
                [14.0, 23.0, 34.0],
                [16.0, 24.5, 34.0],
            ]
        ),
        points_key="beads",
    )

    monkeypatch.setattr(plt, "show", lambda: None)

    fig, ax, sliders = vis_utils.imshow(
        sim,
        points_key="beads",
        points_tolerance=1,
        horizontal_dim="x",
        vertical_dim="y",
    )

    assert list(sliders.keys()) == ["z"]
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"
    np.testing.assert_allclose(
        np.asarray(ax.images[0].get_extent(), dtype=float),
        np.array([28.0, 52.0, 33.5, 18.5]),
    )
    np.testing.assert_allclose(
        ax.collections[0].get_offsets(),
        np.array([[30.0, 20.0]]),
    )

    sliders["z"].set_val(2)
    np.testing.assert_allclose(
        ax.collections[0].get_offsets(),
        np.array([[34.0, 23.0], [34.0, 24.5]]),
    )
    plt.close(fig)


def test_neuroglancer_source_transform_matches_physical_affine():
    sdims = ["y", "x"]
    sim_spacing = {"y": 0.4, "x": 2.0}
    output_spacing = {"y": 1.2, "x": 0.5}
    origin = np.array([12.0, -6.0])
    pixel = np.array([5.0, 8.0])

    theta = np.deg2rad(30)
    linear = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    translation = np.array([3.5, -7.0])
    affine = np.eye(3)
    affine[:2, :2] = linear
    affine[:2, 2] = translation

    ng_affine = vis_utils._affine_to_neuroglancer_source_transform(
        affine,
        sdims=sdims,
        output_spacing=output_spacing,
    )

    sim_spacing_array = np.array([sim_spacing[dim] for dim in sdims])
    output_spacing_array = np.array([output_spacing[dim] for dim in sdims])

    expected_world = linear @ (sim_spacing_array * pixel + origin)
    expected_world += translation

    source_coords = pixel + origin / sim_spacing_array
    ng_linear = (
        ng_affine[:2, :2]
        * sim_spacing_array[None, :]
        / output_spacing_array[:, None]
    )
    ng_output_coords = ng_linear @ source_coords + ng_affine[:2, 2]
    neuroglancer_world = output_spacing_array * ng_output_coords

    np.testing.assert_allclose(neuroglancer_world, expected_world)
    np.testing.assert_allclose(
        ng_affine[:2, 2], translation / output_spacing_array
    )


def test_view_neuroglancer_corrects_spacing_origin_mismatch():
    """
    When in-memory sims have different spacing/origin than the on-disk OME-Zarr,
    generate_neuroglancer_json must include a correction so that the
    pixel → world mapping is consistent with applying the registered affine
    in in-memory physical coordinates.
    """
    sdims = ["y", "x"]
    ndim = len(sdims)

    # On-disk OME-Zarr spacing/origin
    spacing_zarr = {"y": 0.5, "x": 0.5}
    origin_zarr = {"y": 0.0, "x": 0.0}

    # In-memory sim: user changed spacing and origin
    spacing_mem = {"y": 1.0, "x": 2.0}
    origin_mem = {"y": 10.0, "x": -5.0}

    # Registered affine expressed in in-memory physical coordinates
    theta = np.deg2rad(15)
    linear = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    translation = np.array([3.0, -2.0])
    mem_affine = np.eye(ndim + 1)
    mem_affine[:ndim, :ndim] = linear
    mem_affine[:ndim, ndim] = translation

    # Compute the correction that maps zarr physical coords → mem physical coords
    correction = np.eye(ndim + 1)
    for i, dim in enumerate(sdims):
        scale = spacing_mem[dim] / spacing_zarr[dim]
        correction[i, i] = scale
        correction[i, ndim] = origin_mem[dim] - origin_zarr[dim] * scale

    composed_affine = mem_affine @ correction

    ng_affine = vis_utils._affine_to_neuroglancer_source_transform(
        composed_affine,
        sdims=sdims,
        output_spacing=spacing_zarr,
    )

    # Verify for a test pixel that neuroglancer produces the expected world coordinate
    pixel = np.array([3.0, 7.0])

    # Expected world: apply mem_affine to the in-memory physical coordinate of the pixel
    mem_phys = np.array(
        [pixel[i] * spacing_mem[sdims[i]] + origin_mem[sdims[i]] for i in range(ndim)]
    )
    expected_world = linear @ mem_phys + translation

    # Neuroglancer's internal computation (mirrors the formula in
    # test_neuroglancer_source_transform_matches_physical_affine)
    zarr_spacing_arr = np.array([spacing_zarr[dim] for dim in sdims])
    zarr_origin_arr = np.array([origin_zarr[dim] for dim in sdims])
    output_spacing_arr = zarr_spacing_arr  # outputDimensions uses zarr spacing

    source_coords = pixel + zarr_origin_arr / zarr_spacing_arr
    ng_linear = (
        ng_affine[:ndim, :ndim]
        * zarr_spacing_arr[None, :]
        / output_spacing_arr[:, None]
    )
    ng_output_coords = ng_linear @ source_coords + ng_affine[:ndim, ndim]
    neuroglancer_world = output_spacing_arr * ng_output_coords

    np.testing.assert_allclose(neuroglancer_world, expected_world, rtol=1e-10)


def test_view_neuroglancer_spacing_origin_mismatch_integration():
    """
    Integration test: generate_neuroglancer_json produces correct source-transform
    matrices when in-memory sims have spacing/origin different from on-disk OME-Zarr.
    """
    import dask.array as da
    import xarray as xr

    sdims = ["y", "x"]
    ndim = len(sdims)
    tile_size = 10

    spacing_zarr = {"y": 0.5, "x": 0.5}
    origin_zarr = {"y": 0.0, "x": 0.0}
    spacing_mem = {"y": 1.0, "x": 2.0}
    origin_mem = {"y": 10.0, "x": -5.0}

    theta = np.deg2rad(15)
    linear = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    translation = np.array([3.0, -2.0])
    mem_affine_np = np.eye(ndim + 1)
    mem_affine_np[:ndim, :ndim] = linear
    mem_affine_np[:ndim, ndim] = translation
    xaffine = param_utils.affine_to_xaffine(mem_affine_np)

    # Build zarr-backed sim (use dask array so write_sim_to_ome_zarr works)
    sim_zarr = xr.DataArray(
        da.zeros((tile_size, tile_size), chunks=(tile_size, tile_size)),
        dims=sdims,
        coords={
            dim: np.arange(tile_size) * spacing_zarr[dim] + origin_zarr[dim]
            for dim in sdims
        },
    )
    si_utils.set_sim_affine(sim_zarr, xaffine, transform_key="test")

    # Build in-memory sim with different spacing/origin but same registered affine
    sim_mem = xr.DataArray(
        np.zeros((tile_size, tile_size)),
        dims=sdims,
        coords={
            dim: np.arange(tile_size) * spacing_mem[dim] + origin_mem[dim]
            for dim in sdims
        },
    )
    si_utils.set_sim_affine(sim_mem, xaffine, transform_key="test")

    with tempfile.TemporaryDirectory() as data_dir:
        zarr_path = os.path.join(data_dir, "sim.zarr")
        ngff_utils.write_sim_to_ome_zarr(sim_zarr, zarr_path)

        ng_json = vis_utils.generate_neuroglancer_json(
            ome_zarr_paths=[zarr_path],
            ome_zarr_urls=["http://localhost:8000/sim.zarr"],
            sims=[sim_mem],
            transform_key="test",
        )

        # Extract source-transform matrix and output spacing from JSON
        source_transform = ng_json["layers"][0]["source"]["transform"]
        matrix = np.array(source_transform["matrix"])  # shape (len(dims), len(dims)+1)
        output_dims = source_transform["outputDimensions"]
        output_spacing_arr = np.array(
            [output_dims[dim][0] / 1e-6 for dim in sdims]
        )

        # The spatial block lives in the last ndim rows and last ndim+1 cols
        # (linear part in [-ndim:, -ndim-1:-1], translation in [-ndim:, -1])
        ng_linear_block = matrix[-ndim:, -ndim - 1 : -1]
        ng_translation = matrix[-ndim:, -1]

        # Verify pixel → world coordinate
        pixel = np.array([3.0, 7.0])

        mem_phys = np.array(
            [
                pixel[i] * spacing_mem[sdims[i]] + origin_mem[sdims[i]]
                for i in range(ndim)
            ]
        )
        expected_world = linear @ mem_phys + translation

        zarr_spacing_arr = np.array([spacing_zarr[dim] for dim in sdims])
        zarr_origin_arr = np.array([origin_zarr[dim] for dim in sdims])

        ng_linear_scaled = (
            ng_linear_block
            * zarr_spacing_arr[None, :]
            / output_spacing_arr[:, None]
        )
        source_coords = pixel + zarr_origin_arr / zarr_spacing_arr
        ng_output_coords = ng_linear_scaled @ source_coords + ng_translation
        neuroglancer_world = output_spacing_arr * ng_output_coords

        np.testing.assert_allclose(neuroglancer_world, expected_world, rtol=1e-5)


@pytest.mark.parametrize(
    "ndim, N_t, N_c, option",
    [
        (2, 1, 1, ""),
        (2, 2, 1, ""),
        (3, 1, 2, ""),
        (2, None, None, ""),
        (2, None, 1, "different_c_coords"),
    ],
)
def test_ome_zarr_ng(ndim, N_t, N_c, option):
    sims = sample_data.generate_tiled_dataset(
        ndim=ndim,
        overlap=0,
        N_c=N_c if N_c is not None else 1,
        N_t=N_t if N_t is not None else 1,
        tile_size=10,
        tiles_x=2,
        tiles_y=1,
        tiles_z=1,
        spacing_x=0.1,
        spacing_y=0.1,
        spacing_z=2,
    )

    # make sure to also test for the absence of c and t
    if N_c is None:
        for isim in range(len(sims)):
            sims[isim] = sims[isim].drop_vars("c")

    if N_t is None:
        for isim in range(len(sims)):
            sims[isim] = sims[isim].drop_vars("t")

    # test case of different
    if option == "different_c_coords":
        for isim in range(len(sims)):
            sims[isim] = sims[isim].assign_coords(c=[f"channel {isim}"])

    with tempfile.TemporaryDirectory() as data_dir:
        zarr_paths = [
            os.path.join(data_dir, f"sim_{isim}.zarr")
            for isim in range(len(sims))
        ]
        [
            ngff_utils.write_sim_to_ome_zarr(sim, zarr_paths[isim])
            for isim, sim in enumerate(sims)
        ]

        for single_layer in [True, False]:
            ng_json = vis_utils.generate_neuroglancer_json(
                ome_zarr_paths=zarr_paths,
                ome_zarr_urls=[
                    f"https://localhost:8000/{os.path.basename(zp)}"
                    for zp in zarr_paths
                ],
                sims=sims,
                transform_key=io.METADATA_TRANSFORM_KEY,
                single_layer=single_layer,
            )
            assert len(ng_json.keys())

        # test with channel coord
        if option != "different_c_coords":
            ng_json = vis_utils.generate_neuroglancer_json(
                ome_zarr_paths=zarr_paths,
                ome_zarr_urls=[
                    f"https://localhost:8000/{os.path.basename(zp)}"
                    for zp in zarr_paths
                ],
                sims=sims,
                channel_coord=sims[0].coords["c"].values[0],
                transform_key=io.METADATA_TRANSFORM_KEY,
                single_layer=single_layer,
            )
            assert len(ng_json.keys())

        # without sims
        ng_json = vis_utils.generate_neuroglancer_json(
            ome_zarr_paths=zarr_paths,
            ome_zarr_urls=[
                f"https://localhost:8000/{os.path.basename(zp)}"
                for zp in zarr_paths
            ],
        )
        assert len(ng_json.keys())


def test_view_neuroglancer_different_folders(monkeypatch):
    """
    view_neuroglancer must:
    - serve from the common root when zarrs are in different sub-directories
    - use forward slashes in URLs (cross-platform)
    - emit a UserWarning and skip serving when the depth exceeds the max
    """
    import warnings
    import webbrowser

    monkeypatch.setattr(webbrowser, "open", lambda url: None)

    sims = sample_data.generate_tiled_dataset(
        ndim=2,
        overlap=0,
        N_c=1,
        N_t=1,
        tile_size=10,
        tiles_x=2,
        tiles_y=1,
        tiles_z=1,
    )

    with tempfile.TemporaryDirectory() as root_dir:
        # place the two zarrs in different sub-directories
        zarr_paths = []
        for isim, sim in enumerate(sims):
            sub_dir = os.path.join(root_dir, f"subdir_{isim}")
            os.makedirs(sub_dir, exist_ok=True)
            zp = os.path.join(sub_dir, f"sim_{isim}.zarr")
            ngff_utils.write_sim_to_ome_zarr(sim, zp)
            zarr_paths.append(zp)

        served_dirs = []
        monkeypatch.setattr(
            vis_utils, "serve_dir", lambda d, port=8000: served_dirs.append(d)
        )

        generated_urls = []
        original_generate = vis_utils.generate_neuroglancer_json

        def capture_urls(ome_zarr_paths, ome_zarr_urls, **kwargs):
            generated_urls.extend(ome_zarr_urls)
            return original_generate(
                ome_zarr_paths=ome_zarr_paths,
                ome_zarr_urls=ome_zarr_urls,
                **kwargs,
            )

        monkeypatch.setattr(vis_utils, "generate_neuroglancer_json", capture_urls)

        vis_utils.view_neuroglancer(zarr_paths, port=8000)

        common_root = os.path.commonpath(
            [os.path.dirname(os.path.abspath(p)) for p in zarr_paths]
        )

        # server started at the common root
        assert len(served_dirs) == 1
        assert os.path.abspath(served_dirs[0]) == os.path.abspath(common_root)

        # URLs use forward slashes and correct relative paths
        for path, url in zip(zarr_paths, generated_urls):
            rel = os.path.relpath(os.path.abspath(path), common_root).replace(
                os.sep, "/"
            )
            assert url == f"http://localhost:8000/{rel}"
            assert "\\" not in url

    # depth > max: should warn and not serve
    # Use two zarrs whose common ancestor is more than _MAX_SERVE_DEPTH
    # levels above each zarr (e.g. root/a/b/c/d/ and root/e/f/g/h/ -> common=root, depth=5)
    _MAX_SERVE_DEPTH = 3
    with tempfile.TemporaryDirectory() as root_dir:
        deep_zarr_paths = []
        for branch in ["a/b/c/d", "e/f/g/h"]:
            deep_sub = os.path.join(root_dir, *branch.split("/"))
            os.makedirs(deep_sub, exist_ok=True)
            zp = os.path.join(deep_sub, f"sim_{branch.replace('/', '_')}.zarr")
            ngff_utils.write_sim_to_ome_zarr(sims[0], zp)
            deep_zarr_paths.append(zp)

        served_dirs.clear()
        generated_urls.clear()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            vis_utils.view_neuroglancer(deep_zarr_paths, port=8000)

        assert any(issubclass(w.category, UserWarning) for w in caught)
        # serve_dir must NOT have been called
        assert len(served_dirs) == 0
