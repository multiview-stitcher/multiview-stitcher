import networkx as nx
import numpy as np
import xarray as xr

from multiview_stitcher import mv_graph, param_utils
from .utils import get_graph_ndim


def groupwise_resolution_shortest_paths(g_reg, reference_view=None):
    """
    Get final transform parameters by concatenating transforms
    along paths of pairwise affine transformations.

    Output parameters P for each view map coordinates in the view
    into the coordinates of a new coordinate system.

    Note: This function operates on a single connected component
    with single-timepoint transforms.
    """
    if not g_reg.number_of_edges():
        ndim = get_graph_ndim(g_reg)
        params = {
            node: param_utils.identity_transform(ndim) for node in g_reg.nodes
        }
        return params, {"metrics": None, "used_edges": [], "edge_residuals": {}}

    ndim = get_graph_ndim(g_reg)

    # use quality as weight in shortest path (mean over tp currently)
    # make sure that quality is non-negative (shortest path algo requires this)
    quality_min = np.min([g_reg.edges[e]["quality"] for e in g_reg.edges])
    for e in g_reg.edges:
        g_reg.edges[e]["quality_mean"] = np.mean(g_reg.edges[e]["quality"])
        g_reg.edges[e]["quality_mean_inv"] = 1 / (
            (g_reg.edges[e]["quality_mean"] - quality_min) + 0.5
        )

    # get directed graph and invert transforms along edges
    g_reg_di = g_reg.to_directed()
    for e in g_reg.edges:
        sorted_e = tuple(sorted(e))
        g_reg_di.edges[(sorted_e[1], sorted_e[0])][
            "transform"
        ] = param_utils.invert_xparams(g_reg.edges[sorted_e]["transform"])

    node_transforms = {}

    subgraph = g_reg_di

    if reference_view is not None and reference_view in subgraph.nodes:
        ref_node = reference_view
    else:
        ref_node = mv_graph.get_node_with_maximal_edge_weight_sum_from_graph(
            subgraph, weight_key="quality"
        )

    # get shortest paths to ref_node
    paths = {
        n: nx.shortest_path(
            subgraph, target=n, source=ref_node, weight="quality_mean_inv"
        )
        for n in subgraph.nodes
    }

    # Track edges that contribute to any shortest path.
    used_edges = set()

    for n in subgraph.nodes:
        reg_path = paths[n]

        path_pairs = [
            [reg_path[i], reg_path[i + 1]]
            for i in range(len(reg_path) - 1)
        ]
        for pair in path_pairs:
            used_edges.add(tuple(sorted(pair)))

        path_params = param_utils.identity_transform(ndim)

        for pair in path_pairs:
            path_params = param_utils.rebase_affine(
                g_reg_di.edges[(pair[0], pair[1])]["transform"],
                path_params,
            )

        node_transforms[n] = param_utils.invert_xparams(path_params)

    # homogenize dims and coords in node_transforms
    # e.g. if some node's params are missing 't' dimension, add it
    node_transforms = xr.Dataset(data_vars=node_transforms).to_array("node")
    node_transforms = {
        node: node_transforms.sel({"node": node}).drop_vars("node")
        for node in node_transforms.coords["node"].values
    }

    return node_transforms, {
        "metrics": None,
        "used_edges": list(used_edges),
    }
