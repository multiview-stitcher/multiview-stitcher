import copy
import logging
import warnings
from collections.abc import Iterable
from itertools import chain, product
from typing import Union

import dask.array as da
import networkx as nx
import numpy as np
import xarray as xr
from dask import compute, delayed
from scipy.optimize import linprog
from scipy.spatial import (
    ConvexHull,
    HalfspaceIntersection,
    QhullError,
    cKDTree,
)
from skimage.filters import threshold_otsu

from multiview_stitcher import msi_utils, transformation
from multiview_stitcher import spatial_image_utils as si_utils

BoundingBox = dict[str, dict[str, Union[float, int]]]

logger = logging.getLogger(__name__)


class NotEnoughOverlapError(Exception):
    pass


def build_view_adjacency_graph_from_msims(
    msims,
    transform_key,
    overlap_tolerance=None,
    expand=False,
    pairs=None,
):
    """
    Build graph representing view overlap relationships from list of xarrays.

    Used for
      - groupwise registration
      - determining visualization colors

    Parameters
    ----------
    msims : list of MultiscaleSpatialImage
        Input views.
    transform_key : _type_, optional
        Extrinsic coordinate system to consider
    overlap_tolerance : float, optional
        Tolerance for overlap, by default no tolerance
    expand : bool, optional
        Whether to consider views that only touch as overlapping, by default False
    pairs : list of tuples, optional
        List of pairs of view indices to consider, by default None

    Returns
    -------
    networkx.Graph
        Graph containing input images as nodes and edges between overlapping images,
        with overlap area as edge weights.
    """

    g = nx.Graph()
    for iview in range(len(msims)):
        g.add_node(iview)

    sims = [msi_utils.get_sim_from_msim(msim) for msim in msims]

    sdims = si_utils.get_spatial_dims_from_sim(sims[0])
    nsdims = si_utils.get_nonspatial_dims_from_sim(sims[0])

    if len(nsdims):
        sims = [
            si_utils.sim_sel_coords(
                sim, {nsdim: sim.coords[nsdim][0] for nsdim in nsdims}
            )
            for sim in sims
        ]

    stack_propss = [
        si_utils.get_stack_properties_from_sim(
            sim, transform_key=transform_key
        )
        for sim in sims
    ]

    if overlap_tolerance is not None:
        stack_propss = [
            si_utils.extend_stack_props(sp, overlap_tolerance)
            for sp in stack_propss
        ]

    nx.set_node_attributes(
        g, dict(enumerate(stack_propss)), name="stack_props"
    )

    if pairs is None:
        # calculate overlap between pairs of views that are close to each other
        # (closer than the maximum diameter of the views)
        # use scipy cKDTree for performance

        sim_centers = np.array(
            [
                si_utils.get_center_of_sim(sim, transform_key=transform_key)
                for sim in sims
            ]
        )

        sim_diameters = np.array(
            [
                np.linalg.norm(
                    np.array(
                        [
                            stack_props["shape"][dim]
                            * stack_props["spacing"][dim]
                            for dim in sdims
                        ]
                    )
                )
                for stack_props in stack_propss
            ]
        )
        max_diameter = np.max(sim_diameters)

        tree = cKDTree(sim_centers)

        # get all pairs of views that are close to each other
        pairs = []
        for iview in range(len(msims)):
            close_views = tree.query_ball_point(
                sim_centers[iview], max_diameter + 1
            )
            for close_view in close_views:
                if iview == close_view:
                    continue

                pairs.append((iview, close_view))

    overlap_results = []
    for pair in pairs:
        overlap_result = delayed(get_overlap_between_pair_of_stack_props)(
            stack_propss[pair[0]],
            stack_propss[pair[1]],
        )
        overlap_results.append(overlap_result)

    # multithreading doesn't improve performance here (need to check whether
    # this is still true after removing Geometry3D). Using multiprocessing instead.
    # Probably need to confirm here that local dask scheduler doesn't conflict
    # with dask distributed scheduler
    try:
        overlap_results = compute(overlap_results, scheduler="processes")[0]
    except ValueError:
        # if multiprocessing fails, try default scheduler
        # (e.g. when running in JupyterLite)
        overlap_results = compute(overlap_results)[0]

    for pair, overlap_result in zip(pairs, overlap_results):
        overlap_area = overlap_result[0]
        # overlap 0 means one pixel overlap
        if overlap_area > 0:
            g.add_edge(pair[0], pair[1], overlap=overlap_area)

    return g


def get_halfspace_equations_from_stack_props(stack_props):
    """
    Get the halfspace equations from the stack properties.

    Convention:
    x are inside stack_props if for all i:
        ni * x + ci <= 0
    """

    ndim = get_ndim_from_stack_props(stack_props)
    faces = get_faces_from_stack_props(stack_props)
    center = get_center_from_stack_props(stack_props)

    normals = []
    if ndim == 2:
        for face in faces:
            normals.append(
                np.array([-(face[1][1] - face[0][1]), face[1][0] - face[0][0]])
            )

    elif ndim == 3:
        for face in faces:
            normals.append(np.cross(face[1] - face[0], face[2] - face[0]))

    equations = []
    for iface, normal in enumerate(normals):
        normal = normal / np.linalg.norm(normal)
        c = -np.dot(normal, faces[iface][0])
        if np.dot(normal, center) + c > 0:
            # normal = -normal
            normal = -normal
        c = -np.dot(normal, faces[iface][0])
        # print(np.dot(normal, center) + c)
        equations.append(np.concatenate([normal, [c]]))

    return np.array(equations)


def get_overlap_between_pair_of_stack_props(stack_props1, stack_props2):
    """
    Get the overlap between two stack properties.
    """

    halfspace_eq1 = get_halfspace_equations_from_stack_props(stack_props1)
    halfspace_eq2 = get_halfspace_equations_from_stack_props(stack_props2)

    halfspace_eq_combined = np.concatenate([halfspace_eq1, halfspace_eq2])

    # find the feasible point
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.HalfspaceIntersection.html
    norm_vector = np.reshape(
        np.linalg.norm(halfspace_eq_combined[:, :-1], axis=1),
        (halfspace_eq_combined.shape[0], 1),
    )

    c = np.zeros((halfspace_eq_combined.shape[1],))
    c[-1] = -1
    A = np.hstack((halfspace_eq_combined[:, :-1], norm_vector))
    b = -halfspace_eq_combined[:, -1:]
    res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
    feasible_point = res.x[:-1]

    try:
        halfspace_intersection = HalfspaceIntersection(
            halfspace_eq_combined, feasible_point
        )
    except QhullError:
        return -1, None

    intersection_vertices = halfspace_intersection.intersections

    # get the volume of the intersection
    # in 2D this is the area
    volume = ConvexHull(intersection_vertices).volume

    return volume, None


def get_node_with_maximal_edge_weight_sum_from_graph(g, weight_key):
    """
    g: graph containing edges with weight_key weights
    """
    total_node_weights = {
        node: np.sum([g.edges[e][weight_key] for e in g.edges if node in e])
        for node in g.nodes
    }

    ref_node = max(total_node_weights, key=total_node_weights.get)

    return ref_node


def compute_graph_edges(input_g, weight_name="transform", scheduler=None):
    """
    Perform simultaneous compute on all edge attributes with given name
    """

    g = input_g.copy()

    edge_weight_dict = {
        e: g.edges[e][weight_name]
        for e in g.edges
        if weight_name in g.edges[e]
    }

    edge_weight_dict = compute(edge_weight_dict, scheduler=scheduler)[0]

    for e, w in edge_weight_dict.items():
        g.edges[e][weight_name] = w

    return g


def get_nodes_dataset_from_graph(g, node_attribute):
    return xr.Dataset(
        {
            n: g.nodes[n][node_attribute]
            for n in g.nodes
            if node_attribute in g.nodes[n]
        }
    )


def get_faces_from_stack_props(stack_props):
    """
    Get sim stack corners in world coordinates.
    """
    ndim = len(stack_props["origin"])
    sdims = ["z", "y", "x"][-ndim:]

    gv = np.array(list(np.ndindex(tuple([2] * ndim))))

    faces = []
    for iax in range(len(gv[0])):
        for lface in [0, 1]:
            face = gv[np.where(gv[:, iax] == lface)[0]]
            faces.append(face)

    faces = np.array(faces)

    faces = faces * (
        np.array([stack_props["shape"][dim] for dim in sdims]) - 1
    ) * np.array([stack_props["spacing"][dim] for dim in sdims]) + np.array(
        [stack_props["origin"][dim] for dim in sdims]
    )

    if "transform" in stack_props:
        orig_shape = faces.shape
        faces = faces.reshape(-1, ndim)

        affine = stack_props["transform"]
        faces = np.dot(
            affine, np.hstack([faces, np.ones((faces.shape[0], 1))]).T
        ).T[:, :-1]

        faces = faces.reshape(orig_shape)

    return faces


def get_vertices_from_stack_props(stack_props):
    """
    Get sim stack corners in world coordinates.
    """
    ndim = len(stack_props["origin"])
    sdims = ["z", "y", "x"][-ndim:]

    gv = np.array(list(np.ndindex(tuple([2] * ndim))))

    vertices = gv * (
        np.array([stack_props["shape"][dim] for dim in sdims]) - 1
    ) * np.array([stack_props["spacing"][dim] for dim in sdims]) + np.array(
        [stack_props["origin"][dim] for dim in sdims]
    )

    if "transform" in stack_props:
        affine = stack_props["transform"]
        if "t" in affine.dims:
            affine = affine.isel(t=0)
        vertices = transformation.transform_pts(vertices, affine)

    return vertices


def sims_are_far_apart(sim1, sim2, transform_key):
    """ """

    centers = [
        si_utils.get_center_of_sim(sim, transform_key=transform_key)
        for sim in [sim1, sim2]
    ]
    np.linalg.norm(centers[1] - centers[0], axis=0)

    [
        np.linalg.norm(
            np.array(
                [
                    sim.coords[dim][-1] - sim.coords[dim][0]
                    for dim in si_utils.get_spatial_dims_from_sim(sim)
                ]
            )
        )
        for sim in [sim1, sim2]
    ]


def get_spatial_dims_from_stack_properties(stack_props):
    return [
        dim for dim in si_utils.SPATIAL_DIMS if dim in stack_props["origin"]
    ]


def get_center_from_stack_props(stack_props):
    sdims = get_spatial_dims_from_stack_properties(stack_props)
    ndim = len(sdims)

    center = np.array(
        [
            stack_props["origin"][dim]
            + stack_props["spacing"][dim] * (stack_props["shape"][dim] - 1) / 2
            for dim in sdims
        ]
    )

    if "transform" in stack_props:
        affine = stack_props["transform"]
        affine = np.array(affine)
        center = np.concatenate([center, np.ones(1)])
        center = np.matmul(affine, center)[:ndim]

    return center


def get_ndim_from_stack_props(stack_props):
    return len(stack_props["origin"])


def strack_props_are_far_apart(stack_props_1, stack_props_2):
    """ """

    centers = [
        np.mean(get_vertices_from_stack_props(stack_props), axis=0)
        for stack_props in [stack_props_1, stack_props_2]
    ]
    np.linalg.norm(centers[1] - centers[0], axis=0)

    [
        np.linalg.norm(
            np.array(
                [
                    stack_props["origin"][dim]
                    + stack_props["shape"][dim] * stack_props["spacing"][dim]
                    - stack_props["origin"][dim]
                    for dim in stack_props["origin"]
                ]
            )
        )
        for stack_props in [stack_props_1, stack_props_2]
    ]


def points_inside_sim(pts, sim, transform_key):
    """
    Check whether points lie inside of the image domain of sim.
    """

    stack_props = si_utils.get_stack_properties_from_sim(
        sim, transform_key=transform_key
    )

    halfspace_eqs = get_halfspace_equations_from_stack_props(stack_props)

    inside = np.ones(len(pts), dtype=bool)
    for eq in halfspace_eqs:
        inside = inside & (np.dot(pts, eq[:-1]) + eq[-1] <= 0)

    return inside


# def get_greedy_colors(sims, n_colors=2, transform_key=None):
#     """
#     Get colors (indices) from view adjacency graph analysis

#     Idea: use the same logic to determine relevant registration edges
#     """

#     view_adj_graph = build_view_adjacency_graph_from_msims(
#         [msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims],
#         expand=True,
#         transform_key=transform_key,
#     )

#     # thresholds = threshold_multiotsu(overlaps)

#     # strategy: remove edges with overlap values of increasing thresholds until
#     # the graph division into n_colors is successful

#     # modify overlap values
#     # strategy: add a small amount to edge overlap depending on how many edges the nodes it connects have (betweenness?)

#     edge_vals = nx.edge_betweenness_centrality(view_adj_graph)

#     edges = list(view_adj_graph.edges(data=True))
#     for e in edges:
#         edge_vals[tuple(e[:2])] = edge_vals[tuple(e[:2])] + e[2]["overlap"]

#     sorted_unique_vals = sorted(np.unique(list(edge_vals.values())))

#     nx.set_edge_attributes(view_adj_graph, edge_vals, name="edge_val")

#     thresh_ind = 0
#     while 1:
#         colors = nx.coloring.greedy_color(view_adj_graph)
#         if (
#             len(set(colors.values())) <= n_colors
#         ):  # and nx.coloring.equitable_coloring.is_equitable(view_adj_graph, colors):
#             break
#         view_adj_graph.remove_edges_from(
#             [
#                 (a, b)
#                 for a, b, attrs in view_adj_graph.edges(data=True)
#                 if attrs["edge_val"] <= sorted_unique_vals[thresh_ind]
#             ]
#         )
#         thresh_ind += 1

#     greedy_colors = dict(colors.items())

#     return greedy_colors


def get_greedy_colors(sims, n_colors=2, transform_key=None):
    """
    Get colors (indices) from view adjacency graph analysis
    """

    sdims = si_utils.get_spatial_dims_from_sim(sims[0])

    view_adj_graph = build_view_adjacency_graph_from_msims(
        [msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims],
        overlap_tolerance={dim: 1e-5 for dim in sdims},
        transform_key=transform_key,
    )

    # thresholds = threshold_multiotsu(overlaps)

    # strategy: remove edges with overlap values of increasing thresholds until
    # the graph division into n_colors is successful

    # modify overlap values
    # strategy: add a small amount to edge overlap depending on how many edges the nodes it connects have (betweenness?)

    view_adj_graph_pruned, greedy_colors = prune_graph_to_alternating_colors(
        view_adj_graph, n_colors=n_colors
    )

    return greedy_colors


def prune_graph_to_alternating_colors(g, n_colors=2, return_colors=True):
    """
    Prune a graph

    Parameters
    ----------
    g : nx.Graph
        Graph containing edges with overlap values
    n_colors : int, optional
    return_colors : bool, optional

    Returns
    -------
    nx.Graph
        Pruned graph
    """

    # strategy: remove edges with overlap values of increasing thresholds until
    # the graph division into n_colors is successful

    # modify overlap values
    # strategy: add a small amount to edge overlap depending on how many edges the nodes it connects have (betweenness?)
    # TODO: check in which cases this is necessary

    if not len(g.edges):
        if return_colors:
            return g, {n: 0 for n in g.nodes}
        else:
            return g

    g_pruned = copy.deepcopy(g)

    centrality = nx.edge_betweenness_centrality(g)
    max_centrality = max(centrality.values())
    min_centrality = min(centrality.values())

    edges = list(g_pruned.edges(data=True))
    min_overlap = min([e[2]["overlap"] for e in edges])

    # normalize centrality values
    if max_centrality > min_centrality:
        centrality = {
            e: (centrality[e] - min_centrality)
            / (max_centrality - min_centrality)
            * 0.5
            * min_overlap
            for e in centrality
        }

    edge_vals = {}
    for e in edges:
        edge_vals[tuple(e[:2])] = centrality[tuple(e[:2])] + e[2]["overlap"]

    logger.info(edge_vals)

    sorted_unique_vals = sorted(np.unique(list(edge_vals.values())))

    thresh_ind = 0
    while 1:
        colors = nx.coloring.greedy_color(g_pruned)
        if (
            len(set(colors.values())) <= n_colors
        ):  # and nx.coloring.equitable_coloring.is_equitable(g_pruned, colors):
            break
        g_pruned.remove_edges_from(
            [
                (a, b)
                for a, b, attrs in g_pruned.edges(data=True)
                if edge_vals[(a, b)] <= sorted_unique_vals[thresh_ind]
                and min([len(g_pruned.edges(n)) for n in (a, b)]) > 1
            ]
        )
        thresh_ind += 1

    if return_colors:
        return g_pruned, colors
    else:
        return g_pruned


def prune_to_shortest_weighted_paths(g):
    """
    g contains the overlap between views as edge weights

    Strategy:
    - find connected components (CC) in overlap graph
    - for each CC, determine a central reference node
    - for each CC, determine shortest paths to reference node
    - register each pair of views along shortest paths

    """

    g_reg = copy.deepcopy(g)
    g_reg.remove_edges_from(g_reg.edges)

    ccs = list(nx.connected_components(g))

    if np.max([len(cc) for cc in ccs]) < 2:
        raise (NotEnoughOverlapError("No overlap between views/tiles."))
    elif np.min([len(cc) for cc in ccs]) < 2:
        warnings.warn(
            "The following views/tiles have no links with other views:\n%s"
            % list(chain(*[cc for cc in ccs if len(cc) == 1])),
            # % list(np.where([len(cc) == 1 for cc in ccs])[0]),
            UserWarning,
            stacklevel=1,
        )

    for cc in ccs:
        subgraph = g.subgraph(list(cc))

        ref_node = get_node_with_maximal_edge_weight_sum_from_graph(
            subgraph, weight_key="overlap"
        )

        # invert overlap to use as weight in shortest path
        for e in g.edges:
            g.edges[e]["overlap_inv"] = 1 / (
                g.edges[e]["overlap"] + 1
            )  # overlap can be zero

        # get shortest paths to ref_node
        # paths = nx.shortest_path(g_reg, source=ref_node, weight="overlap_inv")
        paths = {
            n: nx.shortest_path(
                g, target=n, source=ref_node, weight="overlap_inv"
            )
            for n in cc
        }

        # get all pairs of views that are connected by a shortest path
        for _, sp in paths.items():
            if len(sp) < 2:
                continue

            # add registration edges
            for i in range(len(sp) - 1):
                g_reg.add_edge(
                    sp[i], sp[i + 1], overlap=g[sp[i]][sp[i + 1]]["overlap"]
                )

    return g_reg


def prune_to_axis_aligned_edges(g, max_angle=0.2):
    """
    Prune away edges that are not orthogonal to image axes.
    This is specifically useful for filtering out diagonal edges on a regular grid of views.
    """

    edges_to_keep = []
    for edge in g.edges:
        verts1 = get_vertices_from_stack_props(g.nodes[edge[0]]["stack_props"])
        verts2 = get_vertices_from_stack_props(g.nodes[edge[1]]["stack_props"])
        ndim = len(verts1[0])

        # get normalized edge vector
        edge_vec = np.mean(verts2, 0) - np.mean(verts1, 0)
        edge_vec = edge_vec / np.linalg.norm(edge_vec)

        # get normalized axes vectors
        # only calculate this for the first view and assume
        # both views have the same axes

        # get non diagonal axes
        vert_grid_inds = np.array(list(np.ndindex(tuple([2] * ndim))))

        ax_vecs = []
        for ind in range(len(vert_grid_inds)):
            if np.sum(vert_grid_inds[ind]) != 1:
                continue
            ax_vec = verts1[ind] - verts1[0]
            ax_vecs.append(ax_vec / np.linalg.norm(ax_vec))

        # calc angle between edge and axes
        for ax_vec in ax_vecs:
            angle = np.arccos(np.abs(np.dot(edge_vec, ax_vec)))
            if angle < max_angle:
                edges_to_keep.append(edge)
                break

    g_pruned = g.edge_subgraph(edges_to_keep)

    # unfreeze graph
    g_pruned = nx.Graph(g_pruned)

    # add nodes that are not connected by any edge
    for node in g.nodes:
        if node not in g_pruned.nodes:
            g_pruned.add_node(node, **g.nodes[node])

    return g_pruned


def filter_edges(g, weight_key="overlap", threshold=None):
    edges_df = nx.to_pandas_edgelist(g)

    if not len(edges_df):
        return g

    if threshold is None:
        threshold = threshold_otsu(np.array(edges_df[weight_key]))

    edges_to_delete_df = edges_df[
        (
            np.array([w.min() for w in edges_df[weight_key]])
            if isinstance(edges_df[weight_key].iloc[0], Iterable)
            else edges_df[weight_key]
        )
        < threshold
    ]

    g_filtered = g.copy()
    g_filtered.remove_edges_from(
        [(r["source"], r["target"]) for ir, r in edges_to_delete_df.iterrows()]
    )

    return g_filtered


def unique_along_axis(a, axis=0):
    """
    Find unique subarrays in axis in N-D array.
    """
    at = np.ascontiguousarray(a.swapaxes(0, axis))
    dt = np.dtype([("values", at.dtype, at.shape[1:])])
    atv = at.view(dt)
    r = np.unique(atv)["values"].swapaxes(0, axis)
    return r


def get_connected_labels(labels, structure):
    """
    Get pairs of connected labels in an n-dimensional input array.
    """
    ndim = labels.ndim
    structure = np.ones((3,) * ndim)

    pairs = np.concatenate(
        [
            (lambda x: x[:, x.all(axis=0) * np.diff(x, axis=0)[0] != 0])(
                np.array(
                    [
                        labels[
                            tuple(
                                slice([0, 1][int(pos > 1)], None)
                                for pos in pos_structure_coord
                            )
                        ],
                        labels[
                            tuple(
                                slice(0, [None, -1][int(pos > 1)])
                                for pos in pos_structure_coord
                            )
                        ],
                    ]
                ).reshape((2, -1))
            )
            for pos_structure_coord in np.array(np.where(structure)).T
            if (min(pos_structure_coord) < 1 or max(pos_structure_coord) < 2)
        ],
        axis=1,
    )

    pairs = unique_along_axis(pairs, axis=1).T
    pairs -= 1

    return pairs


def get_chunk_bbs(
    array_bb: BoundingBox,
    chunksizes: dict[str, Union[int, list[int]]],
) -> list[BoundingBox]:
    """
    Get chunk bounding boxes for all chunks from array bounding box and chunksize.

    Parameters
    ----------
    array_bb : dict
        Array bounding box with keys 'origin' and 'shape', which each are
        dicts containing values for each dimension.
    chunksize : dict[str, Union[int, list[int]]]
        A dict containing for each dimension either a regular chunksize
        or a list of chunk sizes.

    Returns
    -------
    array of dicts
    block_indices
    """

    spatial_dims = sorted(array_bb["origin"].keys())[::-1]
    chunksizes = [chunksizes[dim] for dim in spatial_dims]
    array_shape = [array_bb["shape"][dim] for dim in spatial_dims]
    array_origin = [array_bb["origin"][dim] for dim in spatial_dims]

    normalized_chunks = da.core.normalize_chunks(chunksizes, array_shape)

    block_indices = list(
        product(*(range(len(bds)) for bds in normalized_chunks))
    )
    block_offsets = [np.cumsum((0,) + bds[:-1]) for bds in normalized_chunks]
    block_shapes = list(normalized_chunks)

    chunk_bbs = [
        {
            "origin": {
                dim: array_origin[idim]
                + array_bb["spacing"][dim]
                * block_offsets[idim][block_ind[idim]]
                for idim, dim in enumerate(spatial_dims)
            },
            "shape": {
                dim: block_shapes[idim][block_ind[idim]]
                for idim, dim in enumerate(spatial_dims)
            },
            "spacing": array_bb["spacing"],
        }
        for block_ind in block_indices
    ]

    return chunk_bbs, block_indices


def get_overlap_for_bbs(
    target_bb: BoundingBox,
    query_bbs: BoundingBox,
    param: xr.DataArray,
    additional_extent_in_pixels: dict[str, int] = None,
    tol: float = 1e-6,
):
    """
    Get slices of query bounding boxes that overlap with target bounding box.

    Parameters
    ----------
    target_bb : dict[str, dict[str, Union[int, float]]]
        Target bounding box.
    query_bbs : list[dict[str, dict[str, Union[int, float]]]]
        Query bounding boxes.
    param : xr.DataArray
        Affine transformation parameters mapping query to target.
    overlap_in_pixels : int
        Additional overlap in pixels.

    Returns
    -------

    """

    if additional_extent_in_pixels is None:
        additional_extent_in_pixels = {"z": 0, "y": 0, "x": 0}
    ndim = len(target_bb["origin"])
    spatial_dims = si_utils.SPATIAL_DIMS[-ndim:]

    corners_target = get_vertices_from_stack_props(
        target_bb,
    )

    # project corners into intrinsic coordinate system
    corners_query = transformation.transform_pts(
        corners_target,
        np.linalg.inv(param.data),
    )

    corners_query_min = np.min(corners_query, axis=0)
    corners_query_max = np.max(corners_query, axis=0)

    overlap_bbs = []
    for query_bb in query_bbs:
        backproj_bb_origin = {
            dim: corners_query_min[idim]
            - additional_extent_in_pixels[dim] * query_bb["spacing"][dim]
            for idim, dim in enumerate(spatial_dims)
        }

        backproj_bb_shape = {
            dim: np.ceil(
                (corners_query_max[idim] - corners_query_min[idim])
                / query_bb["spacing"][dim]
            ).astype(int)
            + 1
            + 2 * additional_extent_in_pixels[dim]
            for idim, dim in enumerate(spatial_dims)
        }

        # return None if overlap is outside of query bounding box
        if any(
            backproj_bb_origin[dim] - tol
            > query_bb["origin"][dim]
            + (query_bb["shape"][dim] - 1) * query_bb["spacing"][dim]
            for dim in spatial_dims
        ):
            overlap_bbs.append(None)
            continue

        # return None if overlap is outside of query bounding box
        if any(
            backproj_bb_origin[dim]
            + (backproj_bb_shape[dim] - 1) * query_bb["spacing"][dim]
            < query_bb["origin"][dim] - tol
            for dim in spatial_dims
        ):
            overlap_bbs.append(None)
            continue

        overlap_bb_origin = {
            dim: np.max([backproj_bb_origin[dim], query_bb["origin"][dim]])
            for idim, dim in enumerate(spatial_dims)
        }

        overlap_bb_shape = {
            dim: np.ceil(
                (
                    np.min(
                        [
                            backproj_bb_origin[dim]
                            + (backproj_bb_shape[dim] - 1)
                            * query_bb["spacing"][dim],
                            query_bb["origin"][dim]
                            + (query_bb["shape"][dim] - 1)
                            * query_bb["spacing"][dim],
                        ]
                    )
                    - overlap_bb_origin[dim]
                )
                / query_bb["spacing"][dim]
            ).astype(int)
            + 1
            for idim, dim in enumerate(spatial_dims)
        }

        if np.max([overlap_bb_shape[dim] < 1 for dim in spatial_dims]):
            overlap_bbs.append(None)
            continue

        overlap_bbs.append(
            {
                "origin": overlap_bb_origin,
                "shape": overlap_bb_shape,
                "spacing": query_bb["spacing"],
            }
        )

    return overlap_bbs


def project_bb_along_dim(
    bb: BoundingBox,
    dim: str,
):
    """
    Project bounding box along a dimension.

    Parameters
    ----------
    bb : BoundingBox
        bounding box
    dim : str
        dimension to project along

    Returns
    -------
    BoundingBox
        projected bounding box
    """

    bb = {
        key: {dim2: bb[key][dim2] for dim2 in bb[key] if dim2 != dim}
        for key in bb
    }

    return bb
