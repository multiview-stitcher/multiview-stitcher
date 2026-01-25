import numpy as np
import pytest
import xarray as xr

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


def test_edge_residual_calculation():
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
        g_reg, method="shortest_paths", reference_view=0
    )
    residuals = info["edge_residuals"][0]
    used_edges = {
        tuple(sorted(e)) for e in info["used_edges"][0]
    }
    unused_edges = {
        tuple(sorted(e)) for e in g_reg.edges
    } - used_edges

    for e in used_edges:
        assert np.isclose(residuals[e], 0.0, atol=1e-6)
    for e in unused_edges:
        assert residuals[e] > 1e-5


@pytest.mark.parametrize(
    "method",
    ["shortest_paths", "global_optimization"],
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
