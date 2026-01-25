import copy
import logging

import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr
from skimage.transform import EuclideanTransform, SimilarityTransform

from multiview_stitcher import mv_graph, param_utils, transformation
from multiview_stitcher.transforms import AffineTransform, TranslationTransform
from .utils import get_beads_graph_from_reg_graph, get_graph_ndim

logger = logging.getLogger(__name__)


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
        ndim = get_graph_ndim(g_reg)
        params = {
            node: param_utils.identity_transform(ndim)
            for node in g_reg.nodes
        }
        info_dict = {
            "metrics": None,
            "used_edges": [],
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
    }

    return params, info_dict


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
