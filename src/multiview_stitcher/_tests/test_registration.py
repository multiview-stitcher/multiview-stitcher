import dask.array as da
import numpy as np
import pytest
import xarray as xr
from scipy import ndimage

from multiview_stitcher import (
    io,
    msi_utils,
    mv_graph,
    registration,
    sample_data,
    spatial_image_utils,
)
from multiview_stitcher.io import METADATA_TRANSFORM_KEY


def test_pairwise():
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

    pd = registration.register_pair_of_msims_over_time(
        msims[0],
        msims[1],
        registration_binning={dim: 1 for dim in spatial_dims},
        transform_key=METADATA_TRANSFORM_KEY,
    )

    p = pd.compute()

    assert np.allclose(
        p.sel(t=0),
        np.array(
            [[1.0, 0.0, 1.73333333], [0.0, 1.0, 7.58333333], [0.0, 0.0, 1.0]]
        ),
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


def test_register_graph():
    sims = io.read_mosaic_image_into_list_of_spatial_xarrays(
        sample_data.get_mosaic_sample_data_path()
    )

    sims = [sim.sel(c=sim.coords["c"][0]) for sim in sims]
    msims = [
        msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims
    ]

    g = mv_graph.build_view_adjacency_graph_from_msims(
        msims, transform_key=METADATA_TRANSFORM_KEY
    )

    # g_pairs = registration.get_registration_pair_graph(g)
    g_reg = registration.get_registration_graph_from_overlap_graph(
        g, transform_key=METADATA_TRANSFORM_KEY
    )

    assert max(["transform" in g_reg.edges[e] for e in g_reg.edges])

    assert [
        type(g_reg.edges[e]["transform"].data) == da.core.Array
        for e in g_reg.edges
        if "transform" in g_reg.edges[e]
    ]

    g_reg_computed = mv_graph.compute_graph_edges(
        g_reg, scheduler="single-threaded"
    )

    assert [
        type(g_reg_computed.edges[e]["transform"].data) == np.ndarray
        for e in g_reg_computed.edges
        if "transform" in g_reg_computed.edges[e]
    ]

    # get node parameters
    g_reg_nodes = registration.get_node_params_from_reg_graph(g_reg_computed)

    assert ["transforms" in g_reg_nodes.nodes[n] for n in g_reg_nodes.nodes]


def test_get_stabilization_parameters():
    for ndim in [2, 3]:
        N_t = 10

        im = np.random.randint(0, 100, (5,) * ndim, dtype=np.uint16)
        im = ndimage.zoom(im, [10] * ndim, order=1)

        # simulate random stage drifts
        shifts = (np.random.random((N_t, ndim)) - 0.5) * 30

        # simulate drift
        drift = np.cumsum(np.array([[10] * ndim] * N_t), axis=0)

        tl = []
        for t in range(N_t):
            tl.append(
                ndimage.shift(
                    im, shifts[t] + drift[t], order=1, mode="reflect"
                )
            )
        tl = np.array(tl)

        params_da = registration.get_stabilization_parameters(tl, sigma=1)
        params = params_da.compute()

        assert len(params) == N_t

        assert np.all(
            np.abs(
                np.mean(
                    np.diff(shifts, axis=0) - np.diff(params, axis=0), axis=0
                )
                < np.std(shifts)
            )
            / 3
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
