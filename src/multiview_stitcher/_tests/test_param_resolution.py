import numpy as np
import pytest
import xarray as xr
from scipy.spatial.transform import Rotation
import networkx as nx

from multiview_stitcher import (
    msi_utils,
    mv_graph,
    param_resolution,
    param_utils,
    registration,
    sample_data,
    spatial_image_utils,
)
from multiview_stitcher.io import METADATA_TRANSFORM_KEY


def _rotation_matrix_2d(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def _apply_noise_to_affine(
    affine, ndim, rng, t_sigma, rot_sigma, scale_sigma
):
    linear = affine[:ndim, :ndim]
    trans = affine[:ndim, ndim]

    if rot_sigma > 0:
        if ndim == 2:
            theta = rng.normal(0.0, rot_sigma)
            rot = _rotation_matrix_2d(theta)
        else:
            rotvec = rng.normal(0.0, rot_sigma, size=3)
            rot = Rotation.from_rotvec(rotvec).as_matrix()
    else:
        rot = np.eye(ndim)

    if scale_sigma > 0:
        scale = np.exp(rng.normal(0.0, scale_sigma))
    else:
        scale = 1.0

    noisy = affine.copy()
    noisy[:ndim, :ndim] = (scale * rot) @ linear
    noisy[:ndim, ndim] = trans + rng.normal(0.0, t_sigma, size=ndim)
    return noisy


def _build_synthetic_graph(
    ndim,
    grid_shape,
    mode,
    rng,
    noise_params,
    outlier_fraction=0.0,
):
    g_reg = nx.Graph()
    coords_to_node = {}
    node_to_coords = {}

    idx = 0
    for coords in np.ndindex(grid_shape):
        coords_to_node[coords] = idx
        node_to_coords[idx] = np.array(coords, dtype=float)
        g_reg.add_node(idx)
        idx += 1

    gt_affines = {}
    for node, coords in node_to_coords.items():
        if node == 0:
            affine = np.eye(ndim + 1, dtype=float)
        else:
            translation = coords * 10.0 + rng.normal(0.0, 0.1, size=ndim)
            if mode == "translation":
                rot = np.eye(ndim)
                scale = 1.0
            else:
                if ndim == 2:
                    theta = rng.normal(0.0, 0.03)
                    rot = _rotation_matrix_2d(theta)
                else:
                    rotvec = rng.normal(0.0, 0.03, size=3)
                    rot = Rotation.from_rotvec(rotvec).as_matrix()
                scale = (
                    np.exp(rng.normal(0.0, 0.02))
                    if mode == "similarity"
                    else 1.0
                )
            affine = np.eye(ndim + 1, dtype=float)
            affine[:ndim, :ndim] = scale * rot
            affine[:ndim, ndim] = translation
        gt_affines[node] = affine

    edges = []
    for coords, u in coords_to_node.items():
        for axis in range(ndim):
            neighbor = list(coords)
            neighbor[axis] += 1
            if neighbor[axis] < grid_shape[axis]:
                v = coords_to_node[tuple(neighbor)]
                edges.append((u, v))

    n_outliers = int(len(edges) * outlier_fraction)
    outlier_idx = set()
    if n_outliers:
        outlier_idx = set(
            rng.choice(len(edges), size=n_outliers, replace=False)
        )

    bbox = xr.DataArray(
        np.stack([np.zeros(ndim), np.ones(ndim)]),
        dims=["point_index", "dim"],
    )

    normal_noise = noise_params["normal"]
    outlier_noise = noise_params["outlier"]

    for idx, (u, v) in enumerate(edges):
        affine_uv = np.linalg.inv(gt_affines[v]) @ gt_affines[u]
        if idx in outlier_idx:
            params = outlier_noise
        else:
            params = normal_noise
        affine_uv = _apply_noise_to_affine(
            affine_uv,
            ndim,
            rng,
            params["t_sigma"],
            params["rot_sigma"],
            params["scale_sigma"],
        )
        g_reg.add_edge(
            u,
            v,
            transform=param_utils.affine_to_xaffine(affine_uv),
            quality=1.0,
            overlap=1.0,
            bbox=bbox,
        )

    return g_reg, gt_affines


def _mean_affine_error(params, gt_affines):
    errors = []
    for node, gt_affine in gt_affines.items():
        est = np.asarray(params[node])
        errors.append(np.linalg.norm(est - gt_affine))
    return float(np.mean(errors))


def _make_sample_msims(tiles_x=3, tiles_y=1):
    sims = sample_data.generate_tiled_dataset(
        ndim=2,
        N_t=1,
        N_c=1,
        tile_size=15,
        tiles_x=tiles_x,
        tiles_y=tiles_y,
        tiles_z=1,
        overlap=5,
    )

    sims = [
        spatial_image_utils.sim_sel_coords(sim, {"c": sim.coords["c"][0]})
        for sim in sims
    ]

    return [
        msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims
    ]


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

    msims = _make_sample_msims()

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


@pytest.mark.parametrize(
    "method",
    ["shortest_paths", "global_optimization", "linear_two_pass"],
)
def test_edge_residual_calculation(method):
    """
    Verify that edge residuals are ~0 for edges used by shortest-path
    resolution and non-zero for unused edges on a randomized graph.
    """
    np.random.seed(0)
    msims = _make_sample_msims(tiles_x=3, tiles_y=3)
    g_reg = mv_graph.build_view_adjacency_graph_from_msims(
        msims, transform_key=METADATA_TRANSFORM_KEY
    )

    bbox = xr.DataArray(
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        dims=["point_index", "dim"],
    )
    for e in g_reg.edges:
        g_reg.edges[e]["transform"] = param_utils.affine_to_xaffine(
            param_utils.random_affine(
                ndim=2,
                translation_scale=5,
                rotation_scale=0.01,
                scale_scale=0.02,
            )
        )
        g_reg.edges[e]["quality"] = 1.0
        g_reg.edges[e]["overlap"] = 1.0
        g_reg.edges[e]["bbox"] = bbox

    _, info = param_resolution.groupwise_resolution(
        g_reg, method=method, reference_view=0
    )
    residuals = info["edge_residuals"][0]
    used_edges = {
        tuple(sorted(e)) for e in info["used_edges"][0]
    }
    unused_edges = {
        tuple(sorted(e)) for e in g_reg.edges
    } - used_edges


    if method == "shortest_paths":
        for e in used_edges:
            assert np.isclose(residuals[e], 0.0, atol=1e-6)
        for e in unused_edges:
            assert residuals[e] > 1e-5
    else:
        assert np.min([list(residuals[e] for e in used_edges)]) > 0


@pytest.mark.parametrize(
    "method",
    ["shortest_paths", "global_optimization", "linear_two_pass"],
)
def test_bad_edge_is_not_used(method):
    """
    Verify that an obviously bad edge is excluded from used edges.
    """
    np.random.seed(1)
    msims = _make_sample_msims(tiles_x=3, tiles_y=3)
    g_reg = mv_graph.build_view_adjacency_graph_from_msims(
        msims, transform_key=METADATA_TRANSFORM_KEY
    )

    bbox = xr.DataArray(
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        dims=["point_index", "dim"],
    )
    for e in g_reg.edges:
        g_reg.edges[e]["transform"] = param_utils.affine_to_xaffine(
            param_utils.random_affine(
                ndim=2,
                translation_scale=2,
                rotation_scale=0.01,
                scale_scale=0.02,
            )
        )
        g_reg.edges[e]["quality"] = 1.0
        g_reg.edges[e]["overlap"] = 1.0
        g_reg.edges[e]["bbox"] = bbox

    candidate_edges = [
        e
        for e in g_reg.edges
        if g_reg.degree[e[0]] > 1 and g_reg.degree[e[1]] > 1
    ]
    bad_edge = candidate_edges[0]
    g_reg.edges[bad_edge]["quality"] = 0.01
    bad_transform = param_utils.affine_to_xaffine(np.eye(3))
    bad_transform.data = bad_transform.data.copy()
    bad_transform.data[..., 0, -1] = 100.0
    bad_transform.data[..., 1, -1] = 100.0
    g_reg.edges[bad_edge]["transform"] = bad_transform

    _, info = param_resolution.groupwise_resolution(
        g_reg, method=method, reference_view=0
    )
    used_edges = {
        tuple(sorted(e)) for e in info["used_edges"][0]
    }

    assert tuple(sorted(bad_edge)) not in used_edges


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


def test_linear_two_pass_pruning_improves_error_2d():
    rng = np.random.default_rng(0)
    noise_params = {
        "normal": {"t_sigma": 0.1, "rot_sigma": 0.01, "scale_sigma": 0.005},
        "outlier": {"t_sigma": 5.0, "rot_sigma": 0.4, "scale_sigma": 0.2},
    }
    g_reg, gt_affines = _build_synthetic_graph(
        ndim=2,
        grid_shape=(3, 3),
        mode="similarity",
        rng=rng,
        noise_params=noise_params,
        outlier_fraction=0.25,
    )

    params_all, _ = param_resolution.groupwise_resolution(
        g_reg,
        method="linear_two_pass",
        reference_view=0,
        transform="similarity",
        residual_threshold=np.inf,
        keep_mst=True,
    )
    params_pruned, info = param_resolution.groupwise_resolution(
        g_reg,
        method="linear_two_pass",
        reference_view=0,
        transform="similarity",
        mad_k=2.0,
        keep_mst=True,
    )

    error_all = _mean_affine_error(params_all, gt_affines)
    error_pruned = _mean_affine_error(params_pruned, gt_affines)
    assert error_pruned < error_all

    metrics = info["metrics"]
    assert metrics is not None
    assert (~metrics["kept_pass2"]).any()


def test_linear_two_pass_rigid_3d_accuracy():
    rng = np.random.default_rng(1)
    noise_params = {
        "normal": {"t_sigma": 0.05, "rot_sigma": 0.01, "scale_sigma": 0.0},
        "outlier": {"t_sigma": 0.05, "rot_sigma": 0.01, "scale_sigma": 0.0},
    }
    g_reg, gt_affines = _build_synthetic_graph(
        ndim=3,
        grid_shape=(2, 2, 2),
        mode="rigid",
        rng=rng,
        noise_params=noise_params,
        outlier_fraction=0.0,
    )

    params, _ = param_resolution.groupwise_resolution(
        g_reg,
        method="linear_two_pass",
        reference_view=0,
        transform="rigid",
        residual_threshold=np.inf,
        keep_mst=True,
    )
    error = _mean_affine_error(params, gt_affines)
    assert error < 0.6


def test_linear_two_pass_translation_matches_shortest_paths():
    rng = np.random.default_rng(2)
    noise_params = {
        "normal": {"t_sigma": 0.0, "rot_sigma": 0.0, "scale_sigma": 0.0},
        "outlier": {"t_sigma": 0.0, "rot_sigma": 0.0, "scale_sigma": 0.0},
    }
    g_reg, _ = _build_synthetic_graph(
        ndim=2,
        grid_shape=(3, 1),
        mode="translation",
        rng=rng,
        noise_params=noise_params,
        outlier_fraction=0.0,
    )

    params_linear, _ = param_resolution.groupwise_resolution(
        g_reg,
        method="linear_two_pass",
        reference_view=0,
        transform="translation",
        residual_threshold=np.inf,
    )
    params_paths, _ = param_resolution.groupwise_resolution(
        g_reg,
        method="shortest_paths",
        reference_view=0,
    )

    for node in g_reg.nodes:
        assert np.allclose(
            params_linear[node].data,
            params_paths[node].data,
            atol=1e-6,
        )
