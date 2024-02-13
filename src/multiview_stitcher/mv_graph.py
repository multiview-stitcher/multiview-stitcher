import warnings

import networkx as nx
import numpy as np
import xarray as xr
from dask import compute
from Geometry3D import (
    ConvexPolygon,
    ConvexPolyhedron,
    Point,
    Segment,
)
from skimage.filters import threshold_otsu

# set_eps(get_eps() * 100000) # https://github.com/GouMinghao/Geometry3D/issues/8
# 20230924: set_eps line above created problems
# potentially because it had been called multiple times
# commenting out for now
from multiview_stitcher import msi_utils, spatial_image_utils


class NotEnoughOverlapError(Exception):
    pass


def build_view_adjacency_graph_from_msims(
    msims,
    transform_key,
    expand=False,
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
    expand : bool, optional
        If True, spatial extents of input images is dilated, by default False
    transform_key : _type_, optional
        Extrinsic coordinate system to consider, by default None

    Returns
    -------
    networkx.Graph
        Graph containing input images as nodes and edges between overlapping images,
        with overlap area as edge weights.
    """

    g = nx.Graph()
    for iview in range(len(msims)):
        g.add_node(iview)

    pairs, overlap_results = [], []
    for imsim1, msim1 in enumerate(msims):
        for imsim2, msim2 in enumerate(msims):
            if imsim1 >= imsim2:
                continue

            # overlap_result = delayed(get_overlap_between_pair_of_sims)(
            overlap_result = get_overlap_between_pair_of_sims(
                msi_utils.get_sim_from_msim(msim1),
                msi_utils.get_sim_from_msim(msim2),
                expand=expand,
                transform_key=transform_key,
            )

            pairs.append((imsim1, imsim2))
            overlap_results.append(overlap_result)

    # Threading doesn't improve performance here
    # but actually slows it down, probably because the GIL is not released
    # by pure python Geometry3D code.
    # Multiprocessing should help, but tests don't suggest so.
    # Maybe because in current implementation, probably the full arrays are passed
    # which might get computed. We should consider passing
    # stack properties / boundaries only.
    overlap_results = compute(overlap_results, scheduler="single-threaded")[0]

    for pair, overlap_result in zip(pairs, overlap_results):
        overlap_area = overlap_result[0]
        # overlap 0 means one pixel overlap
        if overlap_area > 0:
            g.add_edge(pair[0], pair[1], overlap=overlap_area)

    return g


def get_overlap_between_pair_of_sims(
    sim1,
    sim2,
    expand=False,
    transform_key=None,
):
    """
    - if there is no overlap, return overlap area of -1
    - if there's a one pixel wide overlap, overlap_area is 0
    - assumes spacing is the same for sim1 and sim2
    """

    if "t" in sim1.dims:
        sim1 = spatial_image_utils.sim_sel_coords(
            sim1, {"t": sim1.coords["t"][0]}
        )

    if "t" in sim2.dims:
        sim2 = spatial_image_utils.sim_sel_coords(
            sim2, {"t": sim2.coords["t"][0]}
        )

    assert spatial_image_utils.get_ndim_from_sim(
        sim1
    ) == spatial_image_utils.get_ndim_from_sim(sim2)

    ndim = spatial_image_utils.get_ndim_from_sim(sim1)

    if ndim == 2:
        intersection_poly_structure = (
            get_intersection_polygon_from_pair_of_sims_2D(
                sim1,
                sim2,
                transform_key=transform_key,
            )
        )

    elif ndim == 3:
        intersection_poly_structure = (
            get_intersection_polyhedron_from_pair_of_sims_3D(
                sim1, sim2, transform_key=transform_key
            )
        )

    spacing = spatial_image_utils.get_spacing_from_sim(sim1, asarray=True)
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


def get_faces_from_sim(sim, transform_key):
    ndim = spatial_image_utils.get_ndim_from_sim(sim)
    gv = np.array(list(np.ndindex(tuple([2] * ndim))))

    faces = []
    for iax in range(len(gv[0])):
        for lface in [0, 1]:
            face = gv[np.where(gv[:, iax] == lface)[0]]
            faces.append(face)

    faces = np.array(faces)

    origin = spatial_image_utils.get_origin_from_sim(sim, asarray=True)
    spacing = spatial_image_utils.get_spacing_from_sim(sim, asarray=True)
    shape = spatial_image_utils.get_shape_from_sim(sim, asarray=True)
    ndim = spatial_image_utils.get_ndim_from_sim(sim)

    faces = faces * (shape - 1) * spacing + origin

    if transform_key is not None:
        orig_shape = faces.shape
        faces = faces.reshape(-1, ndim)

        affine = spatial_image_utils.get_affine_from_sim(
            sim, transform_key=transform_key
        )
        faces = np.dot(
            affine, np.hstack([faces, np.ones((faces.shape[0], 1))]).T
        ).T[:, :-1]

        faces = faces.reshape(orig_shape)

    return faces


def sims_are_far_apart(sim1, sim2, transform_key):
    """ """

    centers = [
        spatial_image_utils.get_center_of_sim(sim, transform_key=transform_key)
        for sim in [sim1, sim2]
    ]
    np.linalg.norm(centers[1] - centers[0], axis=0)

    [
        np.linalg.norm(
            np.array(
                [
                    sim.coords[dim][-1] - sim.coords[dim][0]
                    for dim in spatial_image_utils.get_spatial_dims_from_sim(
                        sim
                    )
                ]
            )
        )
        for sim in [sim1, sim2]
    ]


def get_intersection_polyhedron_from_pair_of_sims_3D(
    sim1, sim2, transform_key
):
    """
    3D intersection is a polyhedron.
    """

    # perform basic check to see if there can be overlap

    if sims_are_far_apart(sim1, sim2, transform_key):
        return None

    cphs = []
    facess = []
    for sim in [sim1, sim2]:
        faces = get_faces_from_sim(sim, transform_key=transform_key)
        cps = ConvexPolyhedron(
            [ConvexPolygon([Point(c) for c in face]) for face in faces]
        )
        facess.append(faces.reshape((-1, 3)))
        cphs.append(cps)

    if min([any((f == facess[0]).all(1)) for f in facess[1]]) and min(
        [any((f == facess[1]).all(1)) for f in facess[0]]
    ):
        return cphs[0]
    else:
        return cphs[0].intersection(cphs[1])


def get_intersection_polygon_from_pair_of_sims_2D(sim1, sim2, transform_key):
    """
    For 2D, the intersection is a polygon. Still three-dimensional, but with z=0.
    """

    if sims_are_far_apart(sim1, sim2, transform_key):
        return None

    cps = [
        get_poly_from_sim(sim, transform_key=transform_key)
        for sim in [sim1, sim2]
    ]

    return cps[0].intersection(cps[1])


def points_inside_sim(pts, sim, transform_key):
    """
    Check whether points lie inside of the image domain of sim.

    Performance could be improved by adding sth similar to `sims_far_apart`.
    """

    ndim = spatial_image_utils.get_ndim_from_sim(sim)
    assert len(pts[0]) == ndim

    sim_domain = get_poly_from_sim(sim, transform_key=transform_key)

    return np.array(
        [sim_domain.intersection(Point(pt)) is not None for pt in pts]
    )


def get_poly_from_sim(sim, transform_key):
    """
    Get Geometry3D.ConvexPolygon or Geometry3D.ConvexPolyhedron
    representing the image domain of sim.
    """

    ndim = spatial_image_utils.get_ndim_from_sim(sim)

    if ndim == 2:
        corners = np.unique(
            get_faces_from_sim(sim, transform_key=transform_key).reshape(
                (-1, 2)
            ),
            axis=0,
        )
        sim_domain = ConvexPolygon([Point([0] + list(c)) for c in corners])

    elif ndim == 3:
        faces = get_faces_from_sim(sim, transform_key=transform_key)
        sim_domain = ConvexPolyhedron(
            [ConvexPolygon([Point(c) for c in face]) for face in faces]
        )

    return sim_domain


def get_greedy_colors(sims, n_colors=2, transform_key=None):
    """
    Get colors (indices) from view adjacency graph analysis

    Idea: use the same logic to determine relevant registration edges
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

    edge_vals = nx.edge_betweenness_centrality(view_adj_graph)

    edges = list(view_adj_graph.edges(data=True))
    for e in edges:
        edge_vals[tuple(e[:2])] = edge_vals[tuple(e[:2])] + e[2]["overlap"]

    sorted_unique_vals = sorted(np.unique(list(edge_vals.values())))

    nx.set_edge_attributes(view_adj_graph, edge_vals, name="edge_val")

    thresh_ind = 0
    while 1:
        colors = nx.coloring.greedy_color(view_adj_graph)
        if (
            len(set(colors.values())) <= n_colors
        ):  # and nx.coloring.equitable_coloring.is_equitable(view_adj_graph, colors):
            break
        view_adj_graph.remove_edges_from(
            [
                (a, b)
                for a, b, attrs in view_adj_graph.edges(data=True)
                if attrs["edge_val"] <= sorted_unique_vals[thresh_ind]
            ]
        )
        thresh_ind += 1

    greedy_colors = dict(colors.items())

    return greedy_colors


def prune_to_shortest_weighted_paths(g):
    """
    g contains the overlap between views as edge weights

    Strategy:
    - find connected components (CC) in overlap graph
    - for each CC, determine a central reference node
    - for each CC, determine shortest paths to reference node
    - register each pair of views along shortest paths

    """

    g_reg = nx.Graph(nodes=g.nodes)

    ccs = list(nx.connected_components(g))

    if len(ccs) > 1:
        warnings.warn(
            """
The provided tiles/views do not globally overlap, instead there
are %s connected components composed of the following tile indices:\n"""
            % (len(ccs))
            + "\n".join([str(list(cc)) for cc in ccs])
            + "\nProceeding without registering between the disconnected components.",
            UserWarning,
            stacklevel=1,
        )
    if np.max([len(cc) for cc in ccs]) < 2:
        raise (NotEnoughOverlapError("Not enough overlap between views."))
    elif np.min([len(cc) for cc in ccs]) < 2:
        warnings.warn(
            "The following views have no overlap with other views:\n%s"
            % list(np.where([len(cc) == 1 for cc in ccs])[0]),
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

    if threshold is None:
        threshold = threshold_otsu(np.array(edges_df["overlap"]))

    edges_to_delete_df = edges_df[edges_df.overlap < threshold]

    g_filtered = g.copy()
    g_filtered.remove_edges_from(
        [(r["source"], r["target"]) for ir, r in edges_to_delete_df.iterrows()]
    )

    return g_filtered
