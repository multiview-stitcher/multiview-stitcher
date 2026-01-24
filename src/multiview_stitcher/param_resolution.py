import copy
import logging

import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr
from skimage.transform import (
    EuclideanTransform,
    SimilarityTransform,
)

from multiview_stitcher import mv_graph, param_utils, transformation
from multiview_stitcher.transforms import AffineTransform, TranslationTransform

logger = logging.getLogger(__name__)

_GROUPWISE_RESOLUTION_METHODS = {}


def register_groupwise_resolution_method(name, resolver):
    """
    Register a groupwise resolution method.

    The resolver is called once per connected component and must implement:
    resolver(g_reg_component_tp, **kwargs) -> (params, info_dict).
    The input graph contains single-timepoint transforms.
    """

    if not callable(resolver):
        raise TypeError("Resolver must be callable.")
    _GROUPWISE_RESOLUTION_METHODS[name] = resolver


def _get_groupwise_resolution_method(method):
    if callable(method):
        return method
    if method in _GROUPWISE_RESOLUTION_METHODS:
        return _GROUPWISE_RESOLUTION_METHODS[method]
    raise ValueError(f"Unknown groupwise optimization method: {method}")


def _get_graph_timepoints(g_reg):
    t_coords = []
    for e in g_reg.edges:
        transform = g_reg.edges[e].get("transform")
        if isinstance(transform, xr.DataArray) and "t" in transform.coords:
            t_coords.extend(list(transform.coords["t"].values))
    return sorted(set(t_coords))


def groupwise_resolution(g_reg, method="global_optimization", **kwargs):
    """
    Resolve global parameters by running a method per connected component
    and timepoint.

    Parameters
    ----------
    method : str or Callable
        Name of a registered method, or a callable implementing the
        component-level, single-timepoint resolver API.
    """

    resolver = _get_groupwise_resolution_method(method)
    if not len(g_reg.edges):
        raise (
            mv_graph.NotEnoughOverlapError(
                "Not enough overlap between views\
        for stitching."
            )
        )

    # if only two views, set reference view to the first view
    # this is compatible with a [fixed, moving] convention
    if "reference_view" not in kwargs and len(g_reg.nodes) == 2:
        kwargs["reference_view"] = min(list(g_reg.nodes))

    params = {node: [] for node in g_reg.nodes}
    info_metrics = []
    used_edges_t0 = set()
    residuals_t0 = {}

    # Resolve per timepoint and connected component, then stitch results.
    t_coords = _get_graph_timepoints(g_reg)

    # Normalize to a single-timepoint loop for uniform handling.
    iter_t_coords = t_coords if t_coords else [None]
    g_reg_t0 = None
    for it, t in enumerate(iter_t_coords):
        g_reg_t = (
            get_reg_graph_with_single_tp_transforms(g_reg, t)
            if t is not None
            else g_reg
        )
        if it == 0:
            g_reg_t0 = g_reg_t
        for icc, cc in enumerate(nx.connected_components(g_reg_t)):
            g_reg_subgraph = g_reg_t.subgraph(list(cc))
            if not g_reg_subgraph.number_of_edges():
                ndim = _get_graph_ndim(g_reg_subgraph)
                cc_params = {
                    node: param_utils.identity_transform(ndim)
                    for node in cc
                }
                cc_info = None
            else:
                cc_params, cc_info = resolver(g_reg_subgraph, **kwargs)
            for node in cc:
                params[node].append(cc_params[node])

            if cc_info is not None:
                # Accumulate metrics and edge usage from the resolver.
                metrics = cc_info.get("metrics")
                if metrics is not None:
                    metrics = metrics.copy()
                    if t is not None:
                        metrics["t"] = [t] * len(metrics)
                    if "icc" not in metrics.columns:
                        metrics["icc"] = [icc] * len(metrics)
                    info_metrics.append(metrics)
                if it == 0:
                    used_edges = cc_info.get("used_edges")
                    if used_edges is not None:
                        used_edges_t0.update(
                            tuple(sorted(e)) for e in used_edges
                        )
                    edge_residuals = cc_info.get("edge_residuals")
                    if edge_residuals is not None:
                        residuals_t0.update(
                            {
                                tuple(sorted(e)): v
                                for e, v in edge_residuals.items()
                            }
                        )

    # Concatenate per-timepoint parameters.
    if t_coords:
        params = {
            node: xr.concat(params[node], dim="t").assign_coords(
                {"t": t_coords}
            )
            for node in params
        }
    else:
        params = {node: params[node][0] for node in params}

    # Build the optimized graph for the first timepoint.
    g_opt_t0 = g_reg_t0.copy()
    if used_edges_t0:
        edges_to_remove = [
            e
            for e in g_opt_t0.edges
            if tuple(sorted(e)) not in used_edges_t0
        ]
        g_opt_t0.remove_edges_from(edges_to_remove)
    for node in g_opt_t0.nodes:
        node_params = params[node]
        if isinstance(node_params, xr.DataArray) and "t" in node_params:
            node_params = node_params.sel({"t": t_coords[0]})
        g_opt_t0.nodes[node]["transform"] = node_params
    for e in g_opt_t0.edges:
        g_opt_t0.edges[e]["residual"] = residuals_t0.get(
            tuple(sorted(e)), np.nan
        )

    info_dict = {
        "metrics": pd.concat(info_metrics) if info_metrics else None,
        "optimized_graph_t0": g_opt_t0,
    }
    return params, info_dict


def _get_graph_ndim(g_reg):
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
        ndim = _get_graph_ndim(g_reg)
        params = {
            node: param_utils.identity_transform(ndim) for node in g_reg.nodes
        }
        return params, {"metrics": None, "used_edges": [], "edge_residuals": {}}

    ndim = _get_graph_ndim(g_reg)

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
        "edge_residuals": {},
    }


def groupwise_resolution_global_optimization(
    g_reg,
    reference_view=None,
    transform="translation",
    max_iter=None,
    rel_tol=None,
    abs_tol=None,
):
    """
    Get final transform parameters by global optimization.

    Output parameters P for each view map coordinates in the view
    into the coordinates of a new coordinate system.

    Strategy:
    - for each pairwise registration, compute virtual pairs of beads
        - fixed view: take bounding box corners
        - moving view: transform fixed view corners using inverse of pairwise transform
    - determine optimal transformations in an iterative manner

    Two loops:
    - outer loop: loop over different sets of edges (start with all edges):
        - set transforms of all nodes to identity
        - perform inner loop
        - determine whether result is good enough
        - if not, remove edges based on criterion
        - repeat
    - inner loop: given a set of edges, optimise the transformations of each node
        - for each node, compute the transform that minimizes the distance between
            its virtual beads and associated virtual beads in overlapping views
        - assign the computed transform to the view

    Terms:
    - "edge residual": mean distance between pair of virtual beads
        associated with a registration edge

    References:
    - https://imagej.net/imagej-wiki-static/SPIM_Registration_Method
    - BigStitcher publication: https://www.nature.com/articles/s41592-019-0501-0#Sec2
      - Supplementary Note 2

    Parameters
    ----------
    g_beads_subgraph : nx.Graph
        Virtual bead graph
    transform : str
        Transformation type ('translation', 'rigid', 'similarity' or 'affine')
    ref_node : int
        Reference node which keeps its transformation fixed
    max_iter : int, optional
        Maximum number of iterations of inner loop
    rel_tol : float, optional
        Convergence criterion for inner loop: relative improvement of max edge residual below which loop stops.
        By default 1e-4.
    abs_tol : float, optional
        Convergence criterion for outer loop: absolute value of max edge residual below which loop stops.
        By default the diagonal of the voxel size (max over tiles).

    Returns
    -------
    dict
        Dictionary containing the final transform parameters for each view
    """

    if not g_reg.number_of_edges():
        ndim = _get_graph_ndim(g_reg)
        params = {
            node: param_utils.identity_transform(ndim)
            for node in g_reg.nodes
        }
        info_dict = {
            "metrics": None,
            "used_edges": [],
            "edge_residuals": {},
        }
        return params, info_dict

    if max_iter is None:
        max_iter = 500
        logger.info("Global optimization: setting max_iter to %s", max_iter)
    if rel_tol is None:
        rel_tol = 1e-4
        logger.info("Global optimization: setting rel_tol to %s", rel_tol)

    ndim = g_reg.edges[list(g_reg.edges)[0]]["transform"].shape[-1] - 1

    # if abs_tol is None, assign multiple of voxel diagonal
    if abs_tol is None:
        abs_tol = np.max(
            [
                1.0
                * np.sum(
                    [
                        v**2
                        for v in g_reg.nodes[n]["stack_props"][
                            "spacing"
                        ].values()
                    ]
                )
                ** 0.5
                for n in g_reg.nodes
            ]
        )
        # log without using f strings
        logger.info("Global optimization: setting abs_tol to %s", abs_tol)

    params = {nodes: None for nodes in g_reg.nodes}
    all_dfs = []
    g_reg_subgraph = g_reg

    if reference_view is not None and reference_view in g_reg_subgraph.nodes:
        ref_node = reference_view
    else:
        ref_node = mv_graph.get_node_with_maximal_edge_weight_sum_from_graph(
            g_reg_subgraph, weight_key="quality"
        )

    # Optimize on the virtual bead graph derived from the pairwise registrations.
    g_beads_subgraph = get_beads_graph_from_reg_graph(
        g_reg_subgraph, ndim=ndim
    )

    cc_params, cc_df, cc_g_opt = optimize_bead_subgraph(
        g_beads_subgraph,
        transform,
        ref_node,
        max_iter,
        rel_tol,
        abs_tol,
    )

    g_opt_t0 = cc_g_opt

    for node in g_reg_subgraph.nodes:
        params[node] = cc_params[node]

    if cc_df is not None:
        all_dfs.append(cc_df)

    all_dfs = [df for df in all_dfs if df is not None]
    df = pd.concat(all_dfs) if len(all_dfs) else None

    info_dict = {
        "metrics": df,
        "used_edges": [tuple(sorted(e)) for e in g_opt_t0.edges],
        "edge_residuals": {
            tuple(sorted(e)): g_opt_t0.edges[e].get("residual", np.nan)
            for e in g_opt_t0.edges
        },
    }

    return params, info_dict


register_groupwise_resolution_method(
    "global_optimization", groupwise_resolution_global_optimization
)
register_groupwise_resolution_method(
    "shortest_paths", groupwise_resolution_shortest_paths
)

def get_reg_graph_with_single_tp_transforms(g_reg, t):
    g_reg_t = g_reg.copy()
    for e in g_reg_t.edges:
        for k, v in g_reg_t.edges[e].items():
            if isinstance(v, xr.DataArray) and "t" in v.coords:
                g_reg_t.edges[e][k] = g_reg_t.edges[e][k].sel({"t": t})
    return g_reg_t


def get_beads_graph_from_reg_graph(g_reg_subgraph, ndim):
    """
    Get a graph with virtual bead pairs as edges and view transforms as node attributes.

    Parameters
    ----------
    g_reg_subgraph : nx.Graph
        Registration graph with single tp transforms

    Returns
    -------
    nx.Graph
    """

    # undirected graph containing virtual bead pairs as edges
    g_beads_subgraph = nx.Graph()
    g_beads_subgraph.add_nodes_from(g_reg_subgraph.nodes)
    for e in g_reg_subgraph.edges:
        sorted_e = tuple(sorted(e))
        bbox_lower, bbox_upper = g_reg_subgraph.edges[e]["bbox"].data
        gv = np.array(list(np.ndindex(tuple([2] * len(bbox_lower)))))
        bbox_vertices = gv * (bbox_upper - bbox_lower) + bbox_lower
        affine = g_reg_subgraph.edges[e]["transform"]
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
            quality=g_reg_subgraph.edges[e]["quality"].data,
            overlap=g_reg_subgraph.edges[e]["overlap"],
        )

    # initialise view transforms with identity transforms
    for node in g_reg_subgraph.nodes:
        g_beads_subgraph.nodes[node][
            "affine"
        ] = param_utils.identity_transform(ndim)

    return g_beads_subgraph


def optimize_bead_subgraph(
    g_beads_subgraph,
    transform,
    ref_node,
    max_iter,
    rel_tol,
    abs_tol,
):
    """
    Optimize the virtual bead graph.

    Two loops:

    - outer loop: loop over different sets of edges:
        - start with all edges
        - determine whether result is good enough
        - if not, remove edges based on criterion
    - inner loop: given a set of edges, optimise the transformations of each node
        - for each node, compute the transform that minimizes the distance between
            its virtual beads and associated virtual beads in overlapping views
        - assign the computed transform to the view

    Terms:
    - "edge residual": mean distance between pair of virtual beads
        associated with a registration edge

    Parameters
    ----------
    g_beads_subgraph : nx.Graph
        Virtual bead graph
    transform : str
        Transformation type ('translation', 'rigid', 'similarity' or 'affine')
    ref_node : int
        Reference node which keeps its transformation fixed
    max_iter : int, optional
        Maximum number of iterations of inner loop
    rel_tol : float, optional
        Convergence criterion for inner loop: relative improvement of max edge residual below which loop stops.
    abs_tol : float, optional
        Convergence criterion for outer loop: absolute value of max edge residual below which loop stops.
    Returns
    -------
    nx.Graph
        Optimized virtual bead graph
    """

    g_beads_subgraph = copy.deepcopy(g_beads_subgraph)

    # this makes node labels directly usable as indices
    # (for optimisation purposes)
    mapping = {n: i for i, n in enumerate(g_beads_subgraph.nodes)}
    inverse_mapping = dict(enumerate(g_beads_subgraph.nodes))

    # relabel nodes
    nx.relabel_nodes(g_beads_subgraph, mapping, copy=False)
    # relabel bead dicts
    for e in g_beads_subgraph.edges:
        g_beads_subgraph.edges[e]["beads"] = {
            mapping[k]: v
            for k, v in g_beads_subgraph.edges[e]["beads"].items()
        }

    # calculate an order of views by descending connectivity / number of links
    centralities = nx.degree_centrality(g_beads_subgraph)
    sorted_nodes = sorted(centralities, key=centralities.get, reverse=True)

    ndim = (
        g_beads_subgraph.nodes[list(g_beads_subgraph.nodes)[0]][
            "affine"
        ].shape[-1]
        - 1
    )

    if transform.lower() == "translation":
        transform_generator = TranslationTransform(dimensionality=ndim)
    elif transform.lower() == "rigid":
        transform_generator = EuclideanTransform(dimensionality=ndim)
    elif transform.lower() == "similarity":
        transform_generator = SimilarityTransform(dimensionality=ndim)
    elif transform.lower() == "affine":
        transform_generator = AffineTransform(dimensionality=ndim)
    else:
        raise ValueError(
            f"Unknown transformation type in parameter resolution: {transform}"
        )

    all_nodes = list(mapping.values())

    new_affines = np.array(
        [
            param_utils.matmul_xparams(
                param_utils.identity_transform(ndim),
                g_beads_subgraph.nodes[n]["affine"],
            ).data
            for n in all_nodes
        ]
    )

    mean_residuals = []
    max_residuals = []

    total_iterations = 0
    # first loop: iterate until max / mean residual ratio is below threshold
    while True:
        # second loop: optimise transformations of each node
        iter_all_residuals = []

        edges = list(g_beads_subgraph.edges)

        if not len(edges):
            break

        node_edges = [list(g_beads_subgraph.edges(n)) for n in all_nodes]

        node_beads = [
            np.concatenate(
                [
                    g_beads_subgraph.edges[e]["beads"][n]
                    for ie, e in enumerate(node_edges[n])
                ],
                axis=0,
            )
            for n in all_nodes
        ]

        node_beads = [
            np.concatenate([nb, np.ones((len(nb), 1))], axis=1)
            for nb in node_beads
        ]

        adj_nodes = [
            [
                n
                for ie, e in enumerate(node_edges[curr_node])
                for n in e
                if n != curr_node
            ]
            for curr_node in all_nodes
        ]

        adj_beads = [
            [
                g_beads_subgraph.edges[e]["beads"][n]
                for ie, e in enumerate(node_edges[curr_node])
                for n in e
                if n != curr_node
            ]
            for curr_node in all_nodes
        ]

        adj_beads = [
            [
                np.concatenate([abb, np.ones((len(abb), 1))], axis=1)
                for abb in ab
            ]
            for ab in adj_beads
        ]

        for iteration in range(max_iter):
            for _icn, curr_node in enumerate(sorted_nodes):
                if not len(node_edges[curr_node]):
                    continue

                node_pts = np.dot(
                    new_affines[curr_node], node_beads[curr_node].T
                ).T[:, :-1]

                adj_pts = np.concatenate(
                    [
                        np.dot(new_affines[an], adj_beads[curr_node][ian].T).T
                        for ian, an in enumerate(adj_nodes[curr_node])
                    ],
                    axis=0,
                )[:, :-1]

                if curr_node != ref_node:
                    transform_generator.estimate(node_pts, adj_pts)
                    transform_generator.residuals(node_pts, adj_pts)

                    new_affines[curr_node] = np.matmul(
                        transform_generator.params,
                        new_affines[curr_node],
                    )

                total_iterations += 1

            # calculate edge residuals
            edge_residuals = {}
            for e in g_beads_subgraph.edges:
                node1, node2 = e
                node1_pts = transformation.transform_pts(
                    g_beads_subgraph.edges[e]["beads"][node1],
                    new_affines[node1],
                )
                node2_pts = transformation.transform_pts(
                    g_beads_subgraph.edges[e]["beads"][node2],
                    new_affines[node2],
                )
                edge_residuals[e] = np.linalg.norm(
                    node1_pts - node2_pts, axis=1
                )

            mean_residuals.append(
                np.mean(
                    [
                        np.mean(edge_residuals[e])
                        for e in g_beads_subgraph.edges
                    ]
                )
            )

            max_residuals.append(
                np.max(
                    [np.max(edge_residuals[e]) for e in g_beads_subgraph.edges]
                )
            )

            iter_all_residuals.append(edge_residuals)

            logger.debug(
                "Glob opt iter %s, node %s, mean residual %s, max residual %s",
                iteration,
                curr_node,
                mean_residuals[-1],
                max_residuals[-1],
            )

            # check for convergence
            if iteration > 5:
                max_rel_change = np.max(
                    [
                        np.abs(
                            (
                                iter_all_residuals[-1][e]
                                - iter_all_residuals[-2][e]
                            )
                            / max_residuals[-1]
                            if max_residuals[-1] > 0
                            else 0
                        )
                        for e in g_beads_subgraph.edges
                    ]
                )

                # check if max relative change is below rel_tol
                if max_rel_change < rel_tol:
                    break

        # keep parameters after one iteration if there are
        # less than two edges
        if len(list(g_beads_subgraph.edges)) < 2:
            break

        edges = list(g_beads_subgraph.edges)
        if max_residuals[-1] < abs_tol:
            edge_to_remove = None
        else:
            edge_residual_values = [
                # (1 / float(g_beads_subgraph.edges[e]["overlap"])) ** 2
                (1 - float(g_beads_subgraph.edges[e]["quality"])) ** 2
                * np.sqrt(np.max(edge_residuals[e]))
                * np.log10(
                    np.max(
                        [len(list(g_beads_subgraph.neighbors(n))) for n in e]
                    )
                )
                for e in edges
            ]

            edge_to_remove = edges[np.argmax(edge_residual_values)]
            residual_order = np.argsort(edge_residual_values)[::-1]
            # find first node which had more than one edge and
            # cutting it would leave its nodes in separate connected components
            candidate_ind = 0
            found = False
            while True:
                edge_to_remove = edges[residual_order[candidate_ind]]
                nodes = list(edge_to_remove)
                tmp_subgraph = copy.deepcopy(g_beads_subgraph)
                tmp_subgraph.remove_edge(*edge_to_remove)
                ccs = list(nx.connected_components(tmp_subgraph))
                cc_ind_node1 = [
                    i for i, cc in enumerate(ccs) if nodes[0] in cc
                ][0]
                if nodes[1] in ccs[cc_ind_node1]:
                    found = True
                    break
                if candidate_ind == len(residual_order) - 1:
                    break
                candidate_ind += 1

            if not found:
                edge_to_remove = None

        logger.debug("Glob opt iter %s", iteration)
        logger.debug(
            "Max and mean residuals: %s \t %s",
            max_residuals[-1],
            mean_residuals[-1],
        )

        if edge_to_remove is not None:
            g_beads_subgraph.remove_edge(*edge_to_remove)

            logger.debug(
                "Removing edge %s and restarting glob opt.", edge_to_remove
            )
        else:
            logger.info(
                "Finished glob opt. Max and mean residuals: %s \t %s",
                max_residuals[-1],
                mean_residuals[-1],
            )
            break

    if total_iterations:
        for n in all_nodes:
            # assign new affines to nodes
            g_beads_subgraph.nodes[n]["affine"] = new_affines[n]

        # assign residuals to edges
        # for n, edge_residual in iter_all_residuals[-1].items():
        for e, residual in edge_residuals.items():
            g_beads_subgraph.edges[e]["residual"] = np.mean(residual)

    # undo node relabeling
    # skip bead dict unrelabeling, as it is not needed
    nx.relabel_nodes(g_beads_subgraph, inverse_mapping, copy=False)

    df = pd.DataFrame(
        {
            "mean_residual": mean_residuals,
            "max_residual": max_residuals,
            "iteration": np.arange(len(mean_residuals)),
        }
    )

    params = {
        node: param_utils.affine_to_xaffine(
            g_beads_subgraph.nodes[node]["affine"]
        )
        for node in g_beads_subgraph.nodes
    }
    return params, df, g_beads_subgraph
