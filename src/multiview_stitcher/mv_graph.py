import copy
import warnings
from collections.abc import Iterable
from itertools import chain, product
from typing import Union

import dask.array as da
import networkx as nx
import numpy as np
import xarray as xr
from dask import compute, delayed
from scipy.spatial import cKDTree
from skimage.filters import threshold_otsu

from multiview_stitcher import msi_utils, transformation
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher.misc_utils import DisableLogger

with DisableLogger():
    from Geometry3D import (
        ConvexPolygon,
        ConvexPolyhedron,
        Point,
        Segment,
    )

BoundingBox = dict[str, dict[str, Union[float, int]]]


class NotEnoughOverlapError(Exception):
    pass


def build_view_adjacency_graph_from_msims(
    msims,
    transform_key,
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
        Extrinsic coordinate system to consider, by default None
    expand : bool, optional
        If True, spatial extents of input images is dilated, by default False
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
            expand=expand,
        )
        overlap_results.append(overlap_result)

    # multithreading doesn't improve performance here, probably because the GIL is
    # not released by pure python Geometry3D code. Using multiprocessing instead.
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


def get_overlap_between_pair_of_stack_props(
    stack_props_1,
    stack_props_2,
    expand=False,
):
    """
    - if there is no overlap, return overlap area of -1
    - if there's a one pixel wide overlap, overlap_area is 0
    - assumes spacing is the same for sim1 and sim2
    """

    ndim = len(stack_props_1["origin"])

    intersection_poly_structure = (
        get_intersection_poly_from_pair_of_stack_props(
            stack_props_1, stack_props_2
        )
    )

    spacing = np.array(
        [stack_props_1["spacing"][dim] for dim in ["z", "y", "x"][-ndim:]]
    )
    small_length = np.min(spacing) / 10.0

    if intersection_poly_structure is None:
        overlap = -1
        intersection_poly_structure = None
        intersection_poly_structure_points = None
    elif isinstance(intersection_poly_structure, Point):
        overlap = small_length**ndim if expand else 0
        p = intersection_poly_structure
        intersection_poly_structure_points = {"z": p.z, "y": p.y, "x": p.x}
    elif isinstance(intersection_poly_structure, Segment):
        if expand:
            overlap = intersection_poly_structure.length() * small_length ** (
                ndim - 1
            )
        else:
            overlap = 0
        intersection_poly_structure_points = [
            {"z": p.z, "y": p.y, "x": p.x}
            for p in [
                intersection_poly_structure.start_point,
                intersection_poly_structure.end_point,
            ]
        ]
    elif isinstance(intersection_poly_structure, ConvexPolygon):
        if ndim == 2:
            overlap = intersection_poly_structure.area()
        elif ndim == 3:
            if expand:
                overlap = (
                    intersection_poly_structure.area()
                    * small_length ** (ndim - 2)
                )
            else:
                overlap = 0
        intersection_poly_structure_points = [
            {"z": p.z, "y": p.y, "x": p.x}
            for p in intersection_poly_structure.points
        ]
    elif isinstance(intersection_poly_structure, ConvexPolyhedron):
        overlap = intersection_poly_structure.volume()
        intersection_poly_structure_points = [
            {"z": p.z, "y": p.y, "x": p.x}
            for p in list(intersection_poly_structure.point_set)
        ]

    return overlap, intersection_poly_structure_points


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
        vertices = transformation.transform_pts(
            vertices, stack_props["transform"]
        )

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


def get_intersection_poly_from_pair_of_stack_props(
    stack_props_1,
    stack_props_2,
):
    """
    3D intersection is a polyhedron.
    """

    # perform basic check to see if there can be overlap

    if strack_props_are_far_apart(stack_props_1, stack_props_2):
        return None

    ndim = len(stack_props_1["origin"])

    cphs = []
    facess = []
    for stack_props in [stack_props_1, stack_props_2]:
        # faces = get_vertices_from_stack_props(stack_props)
        faces = get_faces_from_stack_props(stack_props)
        cps = get_poly_from_stack_props(stack_props)

        facess.append(faces.reshape((-1, ndim)))
        cphs.append(cps)

    if ndim == 3 and (
        min([any((f == facess[0]).all(1)) for f in facess[1]])
        and min([any((f == facess[1]).all(1)) for f in facess[0]])
    ):
        return cphs[0]
    else:
        return cphs[0].intersection(cphs[1])


def points_inside_sim(pts, sim, transform_key):
    """
    Check whether points lie inside of the image domain of sim.

    Performance could be improved by adding sth similar to `sims_far_apart`.
    """

    ndim = si_utils.get_ndim_from_sim(sim)
    assert len(pts[0]) == ndim

    sim_domain = get_poly_from_stack_props(
        si_utils.get_stack_properties_from_sim(
            sim, transform_key=transform_key
        )
    )

    return np.array(
        [sim_domain.intersection(Point(pt)) is not None for pt in pts]
    )


def get_poly_from_stack_props(stack_props):
    """
    Get Geometry3D.ConvexPolygon or Geometry3D.ConvexPolyhedron
    representing the image domain of sim.
    """

    ndim = len(stack_props["origin"])

    if ndim == 2:
        corners = np.unique(
            get_vertices_from_stack_props(stack_props).reshape((-1, 2)),
            axis=0,
        )
        sim_domain = ConvexPolygon([Point([0] + list(c)) for c in corners])

    elif ndim == 3:
        # faces = get_vertices_from_stack_props(stack_props)
        faces = get_faces_from_stack_props(stack_props)
        sim_domain = ConvexPolyhedron(
            [ConvexPolygon([Point(c) for c in face]) for face in faces]
        )

    return sim_domain


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

    view_adj_graph = build_view_adjacency_graph_from_msims(
        [msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims],
        expand=True,
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

    g_pruned = copy.deepcopy(g)

    edge_vals = nx.edge_betweenness_centrality(g)

    edges = list(g_pruned.edges(data=True))
    for e in edges:
        edge_vals[tuple(e[:2])] = edge_vals[tuple(e[:2])] + e[2]["overlap"]

    sorted_unique_vals = sorted(np.unique(list(edge_vals.values())))

    # nx.set_edge_attributes(g, edge_vals, name="edge_val")

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

    g_reg = nx.Graph()
    g_reg.add_nodes_from(g.nodes())

    ccs = list(nx.connected_components(g))

    #     if len(ccs) > 1:
    #         warnings.warn(
    #             """
    # The provided tiles/views are not globally linked, instead there
    # are %s connected components composed of the following tile indices:\n"""
    #             % (len(ccs))
    #             + "\n".join([str(list(cc)) for cc in ccs])
    #             + "\nProceeding without registering between the disconnected components.",
    #             UserWarning,
    #             stacklevel=1,
    #         )

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
