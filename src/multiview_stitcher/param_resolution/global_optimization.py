import copy
import logging

import networkx as nx
import numpy as np
import pandas as pd
from skimage.transform import EuclideanTransform, SimilarityTransform

from multiview_stitcher import mv_graph, param_utils, transformation
from multiview_stitcher._numba_acceleration import get_numba_acceleration
from multiview_stitcher.transforms import AffineTransform, TranslationTransform
from .utils import get_beads_graph_from_reg_graph, get_graph_ndim

logger = logging.getLogger(__name__)


# Lazy-compiled numba kernels (compiled on first use, then cached)
_numba_kernels = None


def _get_numba_kernels():
    """Return (inner_loop_translation, compute_residuals) numba functions.

    Compiled on first call and cached for subsequent calls.
    """
    global _numba_kernels
    if _numba_kernels is not None:
        return _numba_kernels

    import numba as nb

    @nb.njit(cache=True)
    def _inner_loop_translation(
        new_affines,
        node_beads_flat,
        node_beads_offsets,
        adj_beads_flat,
        adj_beads_offsets,
        adj_nodes_flat,
        adj_nodes_offsets,
        sorted_nodes,
        ref_node,
        ndim,
    ):
        """One pass of the per-node translation update (JIT-compiled)."""
        for icn in range(len(sorted_nodes)):
            curr_node = sorted_nodes[icn]
            nb_start = node_beads_offsets[curr_node]
            nb_end = node_beads_offsets[curr_node + 1]
            n_beads = nb_end - nb_start

            if n_beads == 0:
                continue

            affine = new_affines[curr_node]

            # node_pts = (affine @ beads.T).T[:, :-1]
            node_pts = np.empty((n_beads, ndim))
            for b in range(n_beads):
                for d in range(ndim):
                    val = 0.0
                    for k in range(ndim + 1):
                        val += affine[d, k] * node_beads_flat[nb_start + b, k]
                    node_pts[b, d] = val

            # adj_pts: concatenate transformed adjacent beads
            an_start = adj_nodes_offsets[curr_node]
            an_end = adj_nodes_offsets[curr_node + 1]
            n_adj = an_end - an_start

            total_adj = 0
            for ian in range(n_adj):
                ab_s = adj_beads_offsets[an_start + ian]
                ab_e = adj_beads_offsets[an_start + ian + 1]
                total_adj += ab_e - ab_s

            adj_pts = np.empty((total_adj, ndim))
            idx = 0
            for ian in range(n_adj):
                an = adj_nodes_flat[an_start + ian]
                aff_an = new_affines[an]
                ab_s = adj_beads_offsets[an_start + ian]
                ab_e = adj_beads_offsets[an_start + ian + 1]
                for b in range(ab_e - ab_s):
                    for d in range(ndim):
                        val = 0.0
                        for k in range(ndim + 1):
                            val += aff_an[d, k] * adj_beads_flat[ab_s + b, k]
                        adj_pts[idx, d] = val
                    idx += 1

            if curr_node != ref_node:
                # translation = mean(adj_pts - node_pts)
                translation = np.zeros(ndim)
                for b in range(n_beads):
                    for d in range(ndim):
                        translation[d] += adj_pts[b, d] - node_pts[b, d]
                for d in range(ndim):
                    translation[d] /= n_beads

                # new_affines[curr_node] = update_mat @ old_affine
                # For translation update_mat: result[d,k] = old[d,k] + t[d]*old[ndim,k]
                old = new_affines[curr_node].copy()
                for d in range(ndim):
                    for k in range(ndim + 1):
                        new_affines[curr_node, d, k] = (
                            old[d, k] + translation[d] * old[ndim, k]
                        )

    @nb.njit(cache=True)
    def _compute_residuals(
        new_affines,
        edge_b1_h,
        edge_b2_h,
        edge_node1_idx,
        edge_node2_idx,
        ndim,
    ):
        """Compute per-edge, per-bead residual norms (JIT-compiled)."""
        n_edges = edge_b1_h.shape[0]
        n_beads = edge_b1_h.shape[1]
        residuals = np.empty((n_edges, n_beads))

        for ei in range(n_edges):
            aff1 = new_affines[edge_node1_idx[ei]]
            aff2 = new_affines[edge_node2_idx[ei]]
            for b in range(n_beads):
                sq_sum = 0.0
                for d in range(ndim):
                    v1 = 0.0
                    v2 = 0.0
                    for k in range(ndim + 1):
                        v1 += aff1[d, k] * edge_b1_h[ei, b, k]
                        v2 += aff2[d, k] * edge_b2_h[ei, b, k]
                    sq_sum += (v1 - v2) ** 2
                residuals[ei, b] = sq_sum**0.5

        return residuals

    _numba_kernels = (_inner_loop_translation, _compute_residuals)
    return _numba_kernels


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
    g_reg : nx.Graph
        Registration graph for a single connected component and a single timepoint.
        Nodes correspond to views and edges to pairwise registrations between views.
    reference_view : hashable, optional
        Identifier of the reference view which keeps its transformation fixed (typically
        its transform is the identity in the final global coordinate system). If ``None``,
        a default reference view is chosen.
    transform : str
        Transformation type ('translation', 'rigid', 'similarity' or 'affine').
    max_iter : int, optional
        Maximum number of iterations of the inner optimization loop.
    rel_tol : float, optional
        Convergence criterion for the inner loop: relative improvement of the maximum
        edge residual below which the loop stops. By default 1e-4.
    abs_tol : float, optional
        Convergence criterion for the outer loop: absolute value of the maximum edge
        residual below which the loop stops. By default the diagonal of the voxel size
        (max over tiles).

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

    Performance
    -----------
    If numba is installed the inner loop is JIT-compiled, giving ~20x
    speedup for large tile grids.  See :func:`set_numba_acceleration`.

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
    tuple
        (params, df, g_beads_subgraph)
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

    is_translation = transform.lower() == "translation"

    if is_translation:
        transform_generator = None  # inlined below
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

    # Decide acceleration strategy
    use_numba = get_numba_acceleration() and is_translation

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

    if is_translation and not use_numba:
        _update_mat = np.eye(ndim + 1)

    # outer loop: iterate, optionally removing bad edges
    while True:
        edges = list(g_beads_subgraph.edges)

        if not len(edges):
            break

        node_edges = [list(g_beads_subgraph.edges(n)) for n in all_nodes]

        # --- Pack bead data ---
        node_beads = [
            np.concatenate(
                [g_beads_subgraph.edges[e]["beads"][n] for e in node_edges[n]],
                axis=0,
            )
            for n in all_nodes
        ]

        node_beads = [
            np.concatenate([nb, np.ones((len(nb), 1))], axis=1)
            for nb in node_beads
        ]

        adj_nodes = [
            [n for e in node_edges[cn] for n in e if n != cn]
            for cn in all_nodes
        ]

        adj_beads = [
            [
                g_beads_subgraph.edges[e]["beads"][n]
                for e in node_edges[cn]
                for n in e
                if n != cn
            ]
            for cn in all_nodes
        ]

        adj_beads = [
            [
                np.concatenate([abb, np.ones((len(abb), 1))], axis=1)
                for abb in ab
            ]
            for ab in adj_beads
        ]

        # Batched edge bead arrays for residual computation
        n_edges = len(edges)
        edge_node1_idx = np.array([e[0] for e in edges], dtype=np.int64)
        edge_node2_idx = np.array([e[1] for e in edges], dtype=np.int64)

        _b1_list, _b2_list = [], []
        for e in edges:
            b1 = g_beads_subgraph.edges[e]["beads"][e[0]]
            b2 = g_beads_subgraph.edges[e]["beads"][e[1]]
            _b1_list.append(
                np.concatenate([b1, np.ones((len(b1), 1))], axis=1)
            )
            _b2_list.append(
                np.concatenate([b2, np.ones((len(b2), 1))], axis=1)
            )
        edge_b1_h = np.array(_b1_list)
        edge_b2_h = np.array(_b2_list)

        # --- Numba path: pack into flat arrays for JIT kernels ---
        if use_numba:
            _nb_inner, _nb_residuals = _get_numba_kernels()

            _parts, _offsets = [], [0]
            for n in all_nodes:
                _parts.append(node_beads[n])
                _offsets.append(_offsets[-1] + len(node_beads[n]))
            node_beads_flat = (
                np.concatenate(_parts) if _parts else np.empty((0, ndim + 1))
            )
            node_beads_offsets = np.array(_offsets, dtype=np.int64)

            _ab_parts, _an_flat, _an_offsets, _ab_offsets = [], [], [0], [0]
            for cn in all_nodes:
                an_count = 0
                for e in node_edges[cn]:
                    for n in e:
                        if n != cn:
                            _an_flat.append(n)
                            abb = g_beads_subgraph.edges[e]["beads"][n]
                            abb_h = np.concatenate(
                                [abb, np.ones((len(abb), 1))], axis=1
                            )
                            _ab_parts.append(abb_h)
                            _ab_offsets.append(_ab_offsets[-1] + len(abb_h))
                            an_count += 1
                _an_offsets.append(_an_offsets[-1] + an_count)

            adj_beads_flat = (
                np.concatenate(_ab_parts)
                if _ab_parts
                else np.empty((0, ndim + 1))
            )
            adj_beads_offsets = np.array(_ab_offsets, dtype=np.int64)
            adj_nodes_flat = np.array(_an_flat, dtype=np.int64)
            adj_nodes_offsets = np.array(_an_offsets, dtype=np.int64)
            sorted_nodes_arr = np.array(sorted_nodes, dtype=np.int64)
            ref_node_mapped = mapping.get(ref_node, ref_node)

        prev_residuals_flat = None

        # inner loop
        for iteration in range(max_iter):

            if use_numba:
                # --- Numba JIT inner loop ---
                _nb_inner(
                    new_affines,
                    node_beads_flat,
                    node_beads_offsets,
                    adj_beads_flat,
                    adj_beads_offsets,
                    adj_nodes_flat,
                    adj_nodes_offsets,
                    sorted_nodes_arr,
                    ref_node_mapped,
                    ndim,
                )
                total_iterations += len(sorted_nodes)

                # --- Numba JIT residuals ---
                all_residuals = _nb_residuals(
                    new_affines,
                    edge_b1_h,
                    edge_b2_h,
                    edge_node1_idx,
                    edge_node2_idx,
                    ndim,
                )
            else:
                # --- Vectorized numpy inner loop ---
                for curr_node in sorted_nodes:
                    if not len(node_edges[curr_node]):
                        continue

                    node_pts = (
                        new_affines[curr_node] @ node_beads[curr_node].T
                    ).T[:, :-1]

                    adj_pts = np.concatenate(
                        [
                            (new_affines[an] @ adj_beads[curr_node][ian].T).T
                            for ian, an in enumerate(adj_nodes[curr_node])
                        ],
                        axis=0,
                    )[:, :-1]

                    if curr_node != ref_node:
                        if is_translation:
                            translation = np.mean(
                                adj_pts - node_pts, axis=0
                            )
                            _update_mat[:ndim, ndim] = translation
                            new_affines[curr_node] = (
                                _update_mat @ new_affines[curr_node]
                            )
                        else:
                            transform_generator.estimate(node_pts, adj_pts)
                            new_affines[curr_node] = (
                                transform_generator.params
                                @ new_affines[curr_node]
                            )

                    total_iterations += 1

                # --- Batched numpy residuals ---
                affines1 = new_affines[edge_node1_idx]
                affines2 = new_affines[edge_node2_idx]
                pts1 = np.einsum(
                    "eij,ebj->ebi", affines1, edge_b1_h
                )[:, :, :-1]
                pts2 = np.einsum(
                    "eij,ebj->ebi", affines2, edge_b2_h
                )[:, :, :-1]
                all_residuals = np.linalg.norm(pts1 - pts2, axis=2)

            # Metrics
            mean_residuals.append(float(np.mean(all_residuals)))
            max_residuals.append(float(np.max(all_residuals)))

            edge_residuals = {
                e: all_residuals[i] for i, e in enumerate(edges)
            }

            logger.debug(
                "Glob opt iter %s, mean residual %s, max residual %s",
                iteration,
                mean_residuals[-1],
                max_residuals[-1],
            )

            # check for convergence
            if iteration > 5:
                curr_flat = all_residuals.ravel()
                if max_residuals[-1] > 0:
                    max_rel_change = float(
                        np.max(
                            np.abs(curr_flat - prev_residuals_flat)
                            / max_residuals[-1]
                        )
                    )
                else:
                    max_rel_change = 0.0

                # check if max relative change is below rel_tol
                if max_rel_change < rel_tol:
                    break

            prev_residuals_flat = all_residuals.ravel()

        # keep parameters after one iteration if there are less than two edges
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
