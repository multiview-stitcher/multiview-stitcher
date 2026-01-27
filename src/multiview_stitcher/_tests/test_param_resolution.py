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


def _make_grid_msims(tiles_x, tiles_y, tile_size=64, overlap=16):
    sims = sample_data.generate_tiled_dataset(
        ndim=2,
        N_t=1,
        N_c=1,
        tile_size=tile_size,
        tiles_x=tiles_x,
        tiles_y=tiles_y,
        tiles_z=1,
        overlap=overlap,
    )
    sims = [
        spatial_image_utils.sim_sel_coords(
            sim, {"c": sim.coords["c"][0], "t": sim.coords["t"][0]}
        )
        for sim in sims
    ]
    return [msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims]


def _build_ground_truth_from_msims(msims, transform, rng, rot_sigma, scale_sigma):
    gt_params = {}
    bbox = None
    for idx, msim in enumerate(msims):
        sim = msi_utils.get_sim_from_msim(msim)
        origin = spatial_image_utils.get_origin_from_sim(sim, asarray=True)
        spacing = spatial_image_utils.get_spacing_from_sim(sim, asarray=True)
        shape = spatial_image_utils.get_shape_from_sim(sim, asarray=True)
        extent = shape * spacing
        bbox = xr.DataArray(
            np.stack([np.zeros(2), extent]),
            dims=["point_index", "dim"],
        )

        if transform == "translation":
            linear = np.eye(2)
        else:
            theta = rng.normal(0.0, rot_sigma)
            linear = _rotation_matrix_2d(theta)
            if transform == "similarity":
                scale = np.exp(rng.normal(0.0, scale_sigma))
                linear = scale * linear

        affine = np.eye(3, dtype=float)
        affine[:2, :2] = linear
        affine[:2, 2] = origin
        gt_params[idx] = param_utils.affine_to_xaffine(affine)

    return gt_params, bbox


def _add_noisy_pairwise_transforms(
    g_base,
    gt_params,
    bbox,
    rng,
    transform,
    trans_sigma_px=0.25,
    rot_sigma=0.005,
    scale_sigma=0.005,
):
    g_reg = g_base.copy()
    spacing = np.array([1.0, 1.0])
    for edge in g_reg.edges:
        u, v = sorted(edge)
        gt_u = np.asarray(gt_params[u])
        gt_v = np.asarray(gt_params[v])
        pairwise = np.linalg.inv(gt_v) @ gt_u

        noise = np.eye(3, dtype=float)
        if transform in ("rigid", "similarity"):
            theta = rng.normal(0.0, rot_sigma)
            rot = _rotation_matrix_2d(theta)
            scale = 1.0
            if transform == "similarity":
                scale = np.exp(rng.normal(0.0, scale_sigma))
            noise[:2, :2] = scale * rot
        noise[:2, 2] = rng.normal(0.0, trans_sigma_px, size=2) * spacing

        noisy = noise @ pairwise
        g_reg.edges[edge]["transform"] = param_utils.affine_to_xaffine(
            noisy
        )
        g_reg.edges[edge]["quality"] = 1.0
        g_reg.edges[edge]["overlap"] = g_reg.edges[edge].get("overlap", 1.0)
        g_reg.edges[edge]["bbox"] = bbox
    return g_reg


def _rebase_params(params, reference_node):
    ref = np.asarray(params[reference_node])
    inv_ref = np.linalg.inv(ref)
    return {node: inv_ref @ np.asarray(mat) for node, mat in params.items()}


def _rms_component_errors_2d(params, gt_params, reference_node=0):
    params_rel = _rebase_params(params, reference_node)
    gt_rel = _rebase_params(gt_params, reference_node)

    t_errors = []
    r_errors = []
    s_errors = []
    for node in params_rel:
        est = params_rel[node]
        gt = gt_rel[node]

        t_errors.append(np.sum((est[:2, 2] - gt[:2, 2]) ** 2))

        est_theta = np.arctan2(est[1, 0], est[0, 0])
        gt_theta = np.arctan2(gt[1, 0], gt[0, 0])
        r_errors.append((est_theta - gt_theta) ** 2)

        est_scale = np.mean(np.linalg.svd(est[:2, :2], compute_uv=False))
        gt_scale = np.mean(np.linalg.svd(gt[:2, :2], compute_uv=False))
        s_errors.append((est_scale - gt_scale) ** 2)

    return (
        float(np.sqrt(np.mean(t_errors))),
        float(np.sqrt(np.mean(r_errors))),
        float(np.sqrt(np.mean(s_errors))),
    )


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
    [
        "shortest_paths",
        "global_optimization",
        "linear_two_pass",
    ],
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
    [
        "shortest_paths",
        "global_optimization",
        "linear_two_pass",
    ],
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
        "linear_two_pass",
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
        "linear_two_pass",
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


@pytest.mark.parametrize("transform", ["translation", "rigid"])
def test_linear_two_pass_matches_reference(transform):
    rng = np.random.default_rng(2)
    noise_params = {
        "normal": {"t_sigma": 0.0, "rot_sigma": 0.0, "scale_sigma": 0.0},
        "outlier": {"t_sigma": 0.0, "rot_sigma": 0.0, "scale_sigma": 0.0},
    }
    g_reg, gt_affines = _build_synthetic_graph(
        ndim=2,
        grid_shape=(3, 1),
        mode=transform,
        rng=rng,
        noise_params=noise_params,
        outlier_fraction=0.0,
    )

    params_linear, _ = param_resolution.groupwise_resolution(
        g_reg,
        method="linear_two_pass",
        reference_view=0,
        transform=transform,
        # residual_threshold=np.inf,
    )

    # shortest_paths concatenates full affines, so it only matches translation.
    if transform == "translation":
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

    error = _mean_affine_error(params_linear, gt_affines)
    assert error < 1e-2


@pytest.mark.parametrize("transform", ["translation", "rigid"])
@pytest.mark.parametrize("grid_size", [5])
def test_linear_two_pass_accuracy_grid(transform, grid_size):
    rng = np.random.default_rng(3)
    msims = _make_grid_msims(tiles_x=grid_size, tiles_y=grid_size)
    g_base = mv_graph.build_view_adjacency_graph_from_msims(
        msims, transform_key=METADATA_TRANSFORM_KEY
    )
    gt_params, bbox = _build_ground_truth_from_msims(
        msims, transform, rng, rot_sigma=0.01, scale_sigma=0.01
    )
    g_reg = _add_noisy_pairwise_transforms(
        g_base,
        gt_params,
        bbox,
        rng,
        transform,
        trans_sigma_px=0.2,
        rot_sigma=0.01,
        scale_sigma=0.01,
    )

    params, _ = param_resolution.groupwise_resolution(
        g_reg,
        method="linear_two_pass",
        reference_view=0,
        transform=transform,
        # prior_lambda=0,
        # residual_threshold=np.inf,
    )

    t_err, r_err, s_err = _rms_component_errors_2d(params, gt_params)
    assert t_err < 0.5
    if transform in ("rigid", "similarity"):
        assert r_err < 0.05
    if transform == "similarity":
        assert s_err < 0.05
