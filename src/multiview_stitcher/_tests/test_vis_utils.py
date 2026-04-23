import os
import tempfile

import numpy as np
import pytest
from matplotlib import pyplot as plt

import multiview_stitcher.spatial_image_utils as si_utils
from multiview_stitcher import (
    io,
    msi_utils,
    ngff_utils,
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
