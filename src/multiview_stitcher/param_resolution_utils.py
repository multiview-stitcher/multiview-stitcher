import numpy as np
import networkx as nx
import xarray as xr

from multiview_stitcher import param_utils, transformation


def get_graph_ndim(g_reg):
    """Infer dimensionality from transforms or node stack properties."""
    if g_reg.number_of_edges():
        return (
            g_reg.get_edge_data(*list(g_reg.edges())[0])["transform"].shape[-1]
            - 1
        )
    if len(g_reg.nodes):
        node = next(iter(g_reg.nodes))
        stack_props = g_reg.nodes[node].get("stack_props", {})
        if "spacing" in stack_props:
            return len(stack_props["spacing"].values())
    raise ValueError("Cannot determine dimensionality from graph.")


def get_beads_graph_from_reg_graph(g_reg_subgraph, ndim):
    """
    Build a virtual bead graph from a registration graph.

    Each edge stores bead coordinates in both nodes' local frames.
    """
    g_beads_subgraph = nx.Graph()
    g_beads_subgraph.add_nodes_from(g_reg_subgraph.nodes)
    for e in g_reg_subgraph.edges:
        sorted_e = tuple(sorted(e))
        bbox_lower, bbox_upper = g_reg_subgraph.edges[e]["bbox"].data
        gv = np.array(list(np.ndindex(tuple([2] * len(bbox_lower)))))
        bbox_vertices = gv * (bbox_upper - bbox_lower) + bbox_lower
        affine = g_reg_subgraph.edges[e]["transform"]
        if isinstance(affine, xr.DataArray):
            affine = affine.data
        quality = g_reg_subgraph.edges[e].get("quality", 1.0)
        if isinstance(quality, xr.DataArray):
            quality = quality.data
        overlap = g_reg_subgraph.edges[e].get("overlap", 1.0)
        g_beads_subgraph.add_edge(
            sorted_e[0],
            sorted_e[1],
            beads={
                sorted_e[0]: bbox_vertices,
                sorted_e[1]: transformation.transform_pts(
                    bbox_vertices,
                    affine,
                ),
            },
            quality=quality,
            overlap=overlap,
        )
    for node in g_reg_subgraph.nodes:
        g_beads_subgraph.nodes[node]["affine"] = param_utils.identity_transform(
            ndim
        )
    return g_beads_subgraph


def compute_edge_residuals(g_reg, params, ndim=None):
    """Compute RMS bead residuals in physical units for all edges."""
    if not g_reg.number_of_edges():
        return {}
    if ndim is None:
        ndim = get_graph_ndim(g_reg)
    g_beads = get_beads_graph_from_reg_graph(g_reg, ndim=ndim)
    residuals = {}
    for e in g_beads.edges:
        node1, node2 = e
        pts1 = transformation.transform_pts(
            g_beads.edges[e]["beads"][node1], params[node1]
        )
        pts2 = transformation.transform_pts(
            g_beads.edges[e]["beads"][node2], params[node2]
        )
        residuals[tuple(sorted(e))] = float(
            np.sqrt(np.mean(np.sum((pts1 - pts2) ** 2, axis=1)))
        )
    return residuals
