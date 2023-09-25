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

# set_eps(get_eps() * 100000) # https://github.com/GouMinghao/Geometry3D/issues/8
# 20230924: set_eps line above created problems
# potentially because it had been called multiple times
# commenting out for now
from ngff_stitcher import msi_utils, spatial_image_utils


def build_view_adjacency_graph_from_msims(
    msims, expand=False, transform_key=None
):
    """
    Build graph representing view overlap relationships from list of xarrays.
    Will be used for
      - groupwise registration
      - determining visualization colors
    """

    g = nx.Graph()
    for iview, msim in enumerate(msims):
        g.add_node(iview, msim=msim)

    for imsim1, msim1 in enumerate(msims):
        for imsim2, msim2 in enumerate(msims):
            if imsim1 >= imsim2:
                continue

            overlap_area, _ = get_overlap_between_pair_of_sims(
                msi_utils.get_sim_from_msim(msim1),
                msi_utils.get_sim_from_msim(msim2),
                expand=expand,
                transform_key=transform_key,
            )

            # overlap 0 means one pixel overlap
            # if overlap_area > -1:
            if overlap_area > 0:
                g.add_edge(imsim1, imsim2, overlap=overlap_area)

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


def get_node_with_masimal_overlap_from_graph(g):
    """
    g: graph containing edges with 'overlap' weight
    """
    total_node_overlaps = {
        node: np.sum([g.edges[e]["overlap"] for e in g.edges if node in e])
        for node in g.nodes
    }

    ref_node = max(total_node_overlaps, key=total_node_overlaps.get)

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


def get_faces_from_sim(sim, transform_key=None):
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

    # if distance_centers > np.sum(diagonal_lengths) / 2.0:
    #     logging.info("sims are far apart")
    #     return True
    # else:
    #     logging.info(
    #         "sims are close: %02d"
    #         % (100 * distance_centers / (np.sum(diagonal_lengths) / 2.0))
    #     )
    #     return False


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


def get_intersection_polygon_from_pair_of_sims_2D(
    sim1, sim2, transform_key=None
):
    """
    For 2D, the intersection is a polygon. Still three-dimensional, but with z=0.
    """

    if sims_are_far_apart(sim1, sim2, transform_key):
        return None

    cps = []
    for sim in [sim1, sim2]:
        corners = np.unique(
            get_faces_from_sim(sim, transform_key=transform_key).reshape(
                (-1, 2)
            ),
            axis=0,
        )
        # cp = ConvexPolygon([Point([0]+list(c)) for c in corners])
        cp = ConvexPolygon([Point(list(c[::-1]) + [0]) for c in corners])
        cps.append(cp)

    return cps[0].intersection(cps[1])


def points_inside_sim(pts, sim, transform_key=None):
    """
    Check whether points lie inside of the image domain of sim.

    Performance could be improved by adding sth similar to `sims_far_apart`.
    """

    ndim = spatial_image_utils.get_ndim_from_sim(sim)
    assert len(pts[0]) == ndim

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

    return np.array(
        [sim_domain.intersection(Point(pt)) is not None for pt in pts]
    )
