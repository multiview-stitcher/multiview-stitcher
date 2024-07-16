import copy
import logging
import os
import tempfile
import warnings

import dask.array as da
import networkx as nx
import numpy as np
import pandas as pd
import skimage.registration
import xarray as xr
from dask import compute, delayed
from dask.utils import has_keyword
from multiscale_spatial_image import MultiscaleSpatialImage
from scipy import ndimage, stats
from skimage.metrics import structural_similarity
from skimage.transform import (
    EuclideanTransform,
)

from multiview_stitcher.transforms import AffineTransform, TranslationTransform

try:
    import ants
except ImportError:
    ants = None

from multiview_stitcher import (
    msi_utils,
    mv_graph,
    param_utils,
    spatial_image_utils,
    transformation,
    vis_utils,
)

logger = logging.getLogger(__name__)


def apply_recursive_dict(func, d):
    res = {}
    if isinstance(d, dict):
        for k, v in d.items():
            res[k] = apply_recursive_dict(func, v)
    else:
        return func(d)
    return res


def link_quality_metric_func(im0, im1t):
    quality = stats.spearmanr(im0.flatten(), im1t.flatten() - 1).correlation
    return quality


def get_optimal_registration_binning(
    sim1,
    sim2,
    max_total_pixels_per_stack=(400) ** 3,
    overlap_tolerance=None,
):
    """
    Heuristic to find good registration binning.

    - assume inputs contain only spatial dimensions.
    - assume inputs have same spacing
    - assume x and y have same spacing
    - so far, assume orthogonal axes

    Ideas:
      - sample physical space homogeneously
      - don't occupy too much memory

    """

    spatial_dims = spatial_image_utils.get_spatial_dims_from_sim(sim1)
    ndim = len(spatial_dims)
    spacings = [
        spatial_image_utils.get_spacing_from_sim(sim, asarray=False)
        for sim in [sim1, sim2]
    ]

    registration_binning = {dim: 1 for dim in spatial_dims}

    if overlap_tolerance is not None:
        raise (NotImplementedError("overlap_tolerance"))

    overlap = {
        dim: max(sim1.shape[idim], sim2.shape[idim])
        for idim, dim in enumerate(spatial_dims)
    }

    while (
        max(
            [
                np.prod(
                    [
                        overlap[dim]
                        / spacings[isim][dim]
                        / registration_binning[dim]
                        for dim in spatial_dims
                    ]
                )
                for isim in range(2)
            ]
        )
        >= max_total_pixels_per_stack
    ):
        dim_to_bin = np.argmin(
            [
                min([spacings[isim][dim] for isim in range(2)])
                for dim in spatial_dims
            ]
        )

        if ndim == 3 and dim_to_bin == 0:
            registration_binning["z"] = registration_binning["z"] * 2
        else:
            for dim in ["x", "y"]:
                registration_binning[dim] = registration_binning[dim] * 2

        spacings = [
            {
                dim: spacings[isim][dim] * registration_binning[dim]
                for dim in spatial_dims
            }
            for isim in range(2)
        ]

    return registration_binning


def get_overlap_bboxes(
    sim1,
    sim2,
    input_transform_key=None,
    output_transform_key=None,
    extension_in_physical_units=0.0,
):
    """
    Get bounding box(es) of overlap between two spatial images
    in coord system given by input_transform_key, transformed
    into coord system given by output_transform_key (intrinsic
    coordinates if None).

    Return: lower and upper bounds of overlap for both input images
    """

    ndim = spatial_image_utils.get_ndim_from_sim(sim1)

    corners = [
        mv_graph.get_vertices_from_stack_props(
            spatial_image_utils.get_stack_properties_from_sim(
                sim, transform_key=input_transform_key
            )
        ).reshape(-1, ndim)
        for sim in [sim1, sim2]
    ]

    if output_transform_key is None:
        # project corners into intrinsic coordinate system
        corners_intrinsic = np.array(
            [
                [
                    transformation.transform_pts(
                        corners[isim],
                        np.linalg.inv(
                            spatial_image_utils.get_affine_from_sim(
                                sim, transform_key=input_transform_key
                            ).data
                        ),
                    )
                    for isim in range(2)
                ]
                for sim in [sim1, sim2]
            ]
        )
        corners_target_space = corners_intrinsic
    elif output_transform_key == input_transform_key:
        corners_target_space = [corners, corners]
    else:
        raise (NotImplementedError)

    lowers = [
        np.max(np.min(corners_target_space[isim], axis=1), axis=0)
        - extension_in_physical_units
        for isim in range(2)
    ]
    uppers = [
        np.min(np.max(corners_target_space[isim], axis=1), axis=0)
        + extension_in_physical_units
        for isim in range(2)
    ]

    return lowers, uppers


def sims_to_intrinsic_coord_system(
    sim1,
    sim2,
    transform_key,
    overlap_bboxes,
):
    """
    Transform images into intrinsic coordinate system of fixed image.

    Return: transformed spatial images
    """

    spatial_dims = spatial_image_utils.get_spatial_dims_from_sim(sim1)

    reg_sims_b = [sim1, sim2]
    lowers, uppers = overlap_bboxes

    # get images into the same physical space (that of sim1)
    spatial_image_utils.get_ndim_from_sim(reg_sims_b[0])
    spacing = np.max(
        [
            spatial_image_utils.get_spacing_from_sim(sim, asarray=True)
            for sim in reg_sims_b
        ],
        axis=0,
    )

    # transform images into intrinsic coordinate system of fixed image
    affines = [
        sim.attrs["transforms"][transform_key].squeeze().data
        for sim in reg_sims_b
    ]
    transf_affine = np.matmul(np.linalg.inv(affines[1]), affines[0])

    shape = np.floor(np.array(uppers[0] - lowers[0]) / spacing + 1).astype(
        np.uint16
    )

    reg_sims_b_t = [
        transformation.transform_sim(
            sim,
            [None, transf_affine][isim],
            output_stack_properties={
                "origin": {
                    dim: lowers[0][idim]
                    for idim, dim in enumerate(spatial_dims)
                },
                "spacing": {
                    dim: spacing[idim] for idim, dim in enumerate(spatial_dims)
                },
                "shape": {
                    dim: shape[idim] for idim, dim in enumerate(spatial_dims)
                },
            },
        )
        for isim, sim in enumerate(reg_sims_b)
    ]

    # attach transforms
    for _, sim in enumerate(reg_sims_b_t):
        spatial_image_utils.set_sim_affine(
            sim,
            spatial_image_utils.get_affine_from_sim(
                sim1, transform_key=transform_key
            ),
            transform_key=transform_key,
        )

    return reg_sims_b_t[0], reg_sims_b_t[1]


def phase_correlation_registration(fixed_data, moving_data, **kwargs):
    """
    Phase correlation registration using a modified version of skimage's
    phase_cross_correlation function.

    Parameters
    ----------
    fixed_data : array-like
    moving_data : array-like

    Returns
    -------
    dict
        'affine_matrix' : array-like
            Homogeneous transformation matrix.
        'quality' : float
            Quality metric.
    """

    im0 = fixed_data.data
    im1 = moving_data.data
    ndim = im0.ndim

    if "upsample_factor" not in kwargs:
        kwargs["upsample_factor"] = 10 if ndim == 2 else 2

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # strategy: compute phase correlation with and without
        # normalization and keep the one with the highest
        # structural similarity score during manual "disambiguation"
        # (which should be a metric orthogonal to the corr coef)
        shift_candidates = []
        for normalization in ["phase", None]:
            shift_candidates.append(
                skimage.registration.phase_cross_correlation(
                    im0,
                    im1,
                    disambiguate=False,
                    normalization=normalization,
                    **kwargs,
                )[0]
            )

    # disambiguate shift manually
    # there seems to be a problem with the scikit-image implementation
    # of disambiguate_shift, but this needs to be checked

    # assume that the shift along any dimension isn't larger than the overlap
    # in the dimension with smallest overlap
    # e.g. if overlap is 50 pixels in x and 200 pixels in y, assume that
    # the shift along x and y is smaller than 50 pixels
    max_shift_per_dim = np.max([im.shape for im in [im0, im1]])

    data_range = np.max([im0, im1]) - np.min([im0, im1])

    disambiguate_metric_vals = []
    quality_metric_vals = []

    t_candidates = []
    for shift_candidate in shift_candidates:
        for s in np.ndindex(
            tuple([1 if shift_candidate[d] == 0 else 4 for d in range(ndim)])
        ):
            t_candidate = []
            for d in range(ndim):
                if s[d] == 0:
                    t_candidate.append(shift_candidate[d])
                elif s[d] == 1:
                    t_candidate.append(-shift_candidate[d])
                elif s[d] == 2:
                    t_candidate.append(-(shift_candidate[d] - im1.shape[d]))
                elif s[d] == 3:
                    t_candidate.append(-shift_candidate[d] - im1.shape[d])
            if np.max(np.abs(t_candidate)) < max_shift_per_dim:
                t_candidates.append(t_candidate)

    if not len(t_candidates):
        return [np.zeros(ndim)]

    for t_ in t_candidates:
        im1t = ndimage.affine_transform(
            im1 + 1,
            param_utils.affine_from_translation(list(t_)),
            order=1,
            mode="constant",
            cval=0,
        )
        mask = im1t > 0

        if float(np.sum(mask)) / np.prod(im1.shape) < 0.1:
            disambiguate_metric_val = -1
            quality_metric_val = -1
        else:
            mask_slices = tuple(
                [
                    slice(0, im0.shape[idim] - int(np.ceil(t_[idim])))
                    if t_[idim] >= 0
                    else slice(-int(np.ceil(t_[idim])), im0.shape[idim])
                    for idim in range(ndim)
                ]
            )

            # structural similarity seems to be better than
            # correlation for disambiguation (need to solidify this)
            min_shape = np.min(im0[mask_slices].shape)
            ssim_win_size = np.min([7, min_shape - ((min_shape - 1) % 2)])
            if ssim_win_size < 3 or np.max(im1t[mask_slices]) <= 1:
                disambiguate_metric_val = -1
            else:
                disambiguate_metric_val = structural_similarity(
                    im0[mask_slices],
                    im1t[mask_slices] - 1,
                    data_range=data_range,
                    win_size=ssim_win_size,
                )

            # spearman seems to be better than structural_similarity
            # for filtering out bad links between views
            quality_metric_val = link_quality_metric_func(
                im0[mask], im1t[mask] - 1
            )

        disambiguate_metric_vals.append(disambiguate_metric_val)
        quality_metric_vals.append(quality_metric_val)

    argmax_index = np.nanargmax(disambiguate_metric_vals)
    t = t_candidates[argmax_index]

    reg_result = {}
    reg_result["affine_matrix"] = param_utils.affine_from_translation(t)
    reg_result["quality"] = quality_metric_vals[argmax_index]

    # return [t, quality_metric_vals[argmax_index]]
    return reg_result


def get_affine_from_intrinsic_affine(
    data_affine,
    sim_fixed,
    sim_moving,
    transform_key_fixed=None,
    transform_key_moving=None,
):
    """
    Determine transform between extrinsic coordinate systems given
    a transform between intrinsic coordinate systems

    x_f_P = D_to_P_f * x_f_D

    x_f_D = M_D * x_c_D
    x_f_P = M_P * x_c_P
    x_f_W = M_W * x_c_W

    D_to_P_f * x_f_D = M_P * D_to_P_c * x_c_D
    x_f_D = inv(D_to_P_f) * M_P * D_to_P_c * x_c_D
    =>
    M_D = inv(D_to_P_f) * M_P * D_to_P_c
    =>
    D_to_P_f * M_D * inv(D_to_P_c) = M_P

    D_to_W_f * M_D * inv(D_to_W_c) = M_W

    x_f_P = D_to_P_f * x_f_D

    x_f_D = M_D * x_c_D
    x_f_P = M_P * x_c_P
    D_to_P_f * x_f_D = M_P * D_to_P_c * x_c_D
    x_f_D = inv(D_to_P_f) * M_P * D_to_P_c * x_c_D
    =>
    M_D = inv(D_to_P_f) * M_P * D_to_P_c
    =>
    D_to_P_f * M_D * inv(D_to_P_c) = M_P
    """

    if transform_key_fixed is None:
        phys2world_fixed = np.eye(data_affine.shape[0])
    else:
        phys2world_fixed = np.array(
            sim_fixed.attrs["transforms"][transform_key_moving]
        )

    if transform_key_moving is None:
        phys2world_moving = np.eye(data_affine.shape[0])
    else:
        phys2world_moving = np.array(
            sim_moving.attrs["transforms"][transform_key_moving]
        )

    D_to_P_f = np.matmul(
        param_utils.affine_from_translation(
            spatial_image_utils.get_origin_from_sim(sim_moving, asarray=True)
        ),
        np.diag(
            list(
                spatial_image_utils.get_spacing_from_sim(
                    sim_moving, asarray=True
                )
            )
            + [1]
        ),
    )
    P_to_W_f = phys2world_moving
    D_to_W_f = np.matmul(
        P_to_W_f,
        D_to_P_f,
    )

    D_to_P_c = np.matmul(
        param_utils.affine_from_translation(
            spatial_image_utils.get_origin_from_sim(sim_fixed, asarray=True)
        ),
        np.diag(
            list(
                spatial_image_utils.get_spacing_from_sim(
                    sim_fixed, asarray=True
                )
            )
            + [1]
        ),
    )
    P_to_W_c = phys2world_fixed
    D_to_W_c = np.matmul(
        P_to_W_c,
        D_to_P_c,
    )

    M_W = np.matmul(D_to_W_f, np.matmul(data_affine, np.linalg.inv(D_to_W_c)))

    return M_W


def register_pair_of_msims(
    msim1,
    msim2,
    transform_key,
    registration_binning=None,
    overlap_tolerance=0.0,
    pairwise_reg_func=phase_correlation_registration,
    pairwise_reg_func_kwargs=None,
):
    """
    Register the input images containing only spatial dimensions.

    Return: Transform in homogeneous coordinates.

    Parameters
    ----------
    msim1 : MultiscaleSpatialImage
        Fixed image.
    msim2 : MultiscaleSpatialImage
        Moving image.
    transform_key : str, optional
        Extrinsic coordinate system to consider as preregistration.
        This affects the calculation of the overlap region and is passed on to the
        registration func, by default None
    registration_binning : dict, optional
    overlap_tolerance : float, optional
        Extend overlap regions considered for pairwise registration.
        - if 0, the overlap region is the intersection of the bounding boxes.
        - if > 0, the overlap region is the intersection of the bounding boxes
            extended by this value in all spatial dimensions.
        - if None, the full images are used for registration
    pairwise_reg_func : func, optional
        Function used for registration, which is passed as input two spatial images,
        a transform_key and precomputed bounding boxes. Returns a transform in
        homogeneous coordinates. By default phase_correlation_registration
    pairwise_reg_func_kwargs : dict, optional
        Additional keyword arguments passed to the registration function

    Returns
    -------
    xarray.DataArray
        Transform in homogeneous coordinates mapping coordinates from the fixed
        to the moving image.
    """

    if pairwise_reg_func_kwargs is None:
        pairwise_reg_func_kwargs = {}

    spatial_dims = msi_utils.get_spatial_dims(msim1)
    ndim = len(spatial_dims)

    sim1 = msi_utils.get_sim_from_msim(msim1)
    sim2 = msi_utils.get_sim_from_msim(msim2)

    lowers, uppers = get_overlap_bboxes(
        sim1,
        sim2,
        input_transform_key=transform_key,
        output_transform_key=None,
        extension_in_physical_units=overlap_tolerance
        if overlap_tolerance is not None
        else 0.0,
    )

    if overlap_tolerance is not None:
        reg_sims = [
            sim.sel(
                {
                    dim: slice(
                        lowers[isim][idim] - 0.001,
                        uppers[isim][idim] + 0.001,
                    )
                    for idim, dim in enumerate(spatial_dims)
                }
            )
            for isim, sim in enumerate([sim1, sim2])
        ]
    else:
        reg_sims = [sim1, sim2]

    if registration_binning is None:
        registration_binning = get_optimal_registration_binning(
            reg_sims[0], reg_sims[1]
        )

    if (
        registration_binning is not None
        and max(registration_binning.values()) > 1
    ):
        reg_sims_b = [
            sim.coarsen(registration_binning, boundary="trim")
            .mean()
            .astype(sim.dtype)
            for sim in reg_sims
        ]
    else:
        reg_sims_b = reg_sims

    # # Optionally perform CLAHE before registration
    # for i in range(2):
    #     # reg_sims_b[i].data = da.from_delayed(
    #     #     delayed(skimage.exposure.equalize_adapthist)(
    #     #         reg_sims_b[i].data, kernel_size=10, clip_limit=0.02, nbins=2 ** 13),
    #     #     shape=reg_sims_b[i].shape, dtype=float)

    #     reg_sims_b[i].data = da.map_overlap(
    #         skimage.exposure.equalize_adapthist,
    #         reg_sims_b[i].data,
    #         kernel_size=10,
    #         clip_limit=0.02,
    #         nbins=2 ** 13,
    #         depth={idim: 10 for idim, k in enumerate(spatial_dims)},
    #         dtype=float
    #         )

    pairwise_reg_func_has_keywords = {
        keyword: has_keyword(pairwise_reg_func, keyword)
        for keyword in [
            "fixed_origin",
            "moving_origin",
            "fixed_spacing",
            "moving_spacing",
            "initial_affine",
        ]
    }

    if not np.any(list(pairwise_reg_func_has_keywords.values())):
        registration_func_space = "pixel_space"

        sims_pixel_space = sims_to_intrinsic_coord_system(
            reg_sims_b[0],
            reg_sims_b[1],
            transform_key=transform_key,
            overlap_bboxes=(lowers, uppers),
        )

        fixed_data = sims_pixel_space[0].data
        moving_data = sims_pixel_space[1].data

    elif np.all(list(pairwise_reg_func_has_keywords.values())):
        registration_func_space = "physical_space"

        fixed_data = reg_sims_b[0].data
        moving_data = reg_sims_b[1].data

        for isim, sim in enumerate(reg_sims_b):
            prefix = ["fixed", "moving"][isim]
            pairwise_reg_func_kwargs[
                "%s_origin" % prefix
            ] = spatial_image_utils.get_origin_from_sim(sim)
            pairwise_reg_func_kwargs[
                "%s_spacing" % prefix
            ] = spatial_image_utils.get_spacing_from_sim(sim)

        # obtain initial transform parameters
        affines = [
            spatial_image_utils.get_affine_from_sim(
                sim, transform_key=transform_key
            )
            .squeeze()
            .data
            for sim in reg_sims_b
        ]
        initial_affine = np.matmul(np.linalg.inv(affines[1]), affines[0])
        pairwise_reg_func_kwargs[
            "initial_affine"
        ] = param_utils.affine_to_xaffine(initial_affine)

    else:
        raise ValueError("Unknown registration function signature")

    param_dict_d = delayed(pairwise_reg_func, nout=1)(
        fixed_data=xr.DataArray(fixed_data, dims=spatial_dims),
        moving_data=xr.DataArray(moving_data, dims=spatial_dims),
        **pairwise_reg_func_kwargs,
    )

    affine = da.from_delayed(
        delayed(lambda x: np.array(x["affine_matrix"]))(param_dict_d),
        shape=(ndim + 1, ndim + 1),
        dtype=float,
    )

    quality = da.from_delayed(
        delayed(lambda x: x["quality"])(param_dict_d),
        shape=(),
        dtype=float,
    )

    if registration_func_space == "pixel_space":
        affine_phys = get_affine_from_intrinsic_affine(
            data_affine=affine,
            sim_fixed=sims_pixel_space[0],
            sim_moving=sims_pixel_space[1],
            transform_key_fixed=transform_key,
            transform_key_moving=transform_key,
        )
    elif registration_func_space == "physical_space":
        affine_phys = np.matmul(
            affines[1], np.matmul(affine, np.linalg.inv(affines[0]))
        )

    param_ds = xr.Dataset(
        data_vars={
            "transform": param_utils.affine_to_xaffine(affine_phys),
            "quality": xr.DataArray(quality),
        }
    )

    # attach bbox in physical coordinates
    lowers_phys, uppers_phys = get_overlap_bboxes(
        sim1,
        sim2,
        input_transform_key=transform_key,
        output_transform_key=transform_key,
    )

    param_ds = param_ds.assign(
        {
            "bbox": xr.DataArray(
                [lowers_phys[0], uppers_phys[0]], dims=["point_index", "dim"]
            )
        }
    )

    return param_ds


def register_pair_of_msims_over_time(
    msim1,
    msim2,
    transform_key,
    registration_binning=None,
    overlap_tolerance=0.0,
    pairwise_reg_func=phase_correlation_registration,
    pairwise_reg_func_kwargs=None,
):
    """
    Apply register_pair_of_msims to each time point of the input images.
    """

    if pairwise_reg_func_kwargs is None:
        pairwise_reg_func_kwargs = {}

    msim1 = msi_utils.ensure_dim(msim1, "t")
    msim2 = msi_utils.ensure_dim(msim2, "t")

    sim1 = msi_utils.get_sim_from_msim(msim1)

    # suppress pandas future warning occuring within xarray.concat
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)

        xp = xr.concat(
            [
                register_pair_of_msims(
                    msi_utils.multiscale_sel_coords(msim1, {"t": t}),
                    msi_utils.multiscale_sel_coords(msim2, {"t": t}),
                    transform_key=transform_key,
                    registration_binning=registration_binning,
                    pairwise_reg_func=pairwise_reg_func,
                    pairwise_reg_func_kwargs=pairwise_reg_func_kwargs,
                    overlap_tolerance=overlap_tolerance,
                )
                for t in sim1.coords["t"].values
            ],
            dim="t",
        )

    xp = xp.assign_coords({"t": sim1.coords["t"].values})

    return xp


def prune_view_adjacency_graph(
    g,
    method=None,
):
    """
    Prune the view adjacency graph
    (i.e. to edges that will be used for registration).

    Available methods:
    - 'shortest_paths_overlap_weighted':
        Prune to shortest paths in overlap graph
        (weighted by overlap).
    - 'otsu_threshold_on_overlap':
        Prune to edges with overlap above Otsu threshold.
        This works well for regular grid arrangements, as
        diagonal edges will be pruned.
    """
    if not len(g.edges):
        raise (
            mv_graph.NotEnoughOverlapError(
                "Not enough overlap between views\
        for stitching."
            )
        )

    if method is None:
        return g

    if method == "alternating_pattern":
        return mv_graph.prune_graph_to_alternating_colors(
            g, return_colors=False
        )

    if method == "shortest_paths_overlap_weighted":
        return mv_graph.prune_to_shortest_weighted_paths(g)

    elif method == "otsu_threshold_on_overlap":
        return mv_graph.filter_edges(g)

    else:
        raise ValueError(f"Unknown graph pruning method: {method}")


def groupwise_resolution(g_reg, method="global_optimization", **kwargs):
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

    if method == "global_optimization":
        return groupwise_resolution_global_optimization(g_reg, **kwargs)
    elif method == "shortest_paths":
        return groupwise_resolution_shortest_paths(g_reg, **kwargs)
    else:
        raise ValueError(f"Unknown groupwise optimization method: {method}")


def groupwise_resolution_shortest_paths(g_reg, reference_view=None):
    """
    Get final transform parameters by concatenating transforms
    along paths of pairwise affine transformations.

    Output parameters P for each view map coordinates in the view
    into the coordinates of a new coordinate system.
    """

    # ndim = msi_utils.get_ndim(g_reg.nodes[list(g_reg.nodes)[0]]["msim"])
    ndim = (
        g_reg.get_edge_data(*list(g_reg.edges())[0])["transform"].shape[-1] - 1
    )

    # use quality as weight in shortest path (mean over tp currently)
    for e in g_reg.edges:
        g_reg.edges[e]["quality_mean"] = np.mean(g_reg.edges[e]["quality"])
        g_reg.edges[e]["quality_mean_inv"] = 1 / (
            g_reg.edges[e]["quality_mean"] + 0.5
        )

    # get directed graph and invert transforms along edges

    g_reg_di = g_reg.to_directed()
    for e in g_reg.edges:
        sorted_e = tuple(sorted(e))
        g_reg_di.edges[(sorted_e[1], sorted_e[0])][
            "transform"
        ] = param_utils.invert_xparams(g_reg.edges[sorted_e]["transform"])

    ccs = list(nx.connected_components(g_reg))

    node_transforms = {}

    for cc in ccs:
        subgraph = g_reg_di.subgraph(list(cc))

        if reference_view is not None and reference_view in cc:
            ref_node = reference_view
        else:
            ref_node = (
                mv_graph.get_node_with_maximal_edge_weight_sum_from_graph(
                    subgraph, weight_key="quality"
                )
            )

        # get shortest paths to ref_node
        # paths = nx.shortest_path(g_reg, source=ref_node, weight="overlap_inv")
        paths = {
            n: nx.shortest_path(
                subgraph, target=n, source=ref_node, weight="quality_mean_inv"
            )
            for n in cc
        }

        for n in subgraph.nodes:
            # reg_path = g_reg.nodes[n]["reg_path"]
            reg_path = paths[n]

            path_pairs = [
                [reg_path[i], reg_path[i + 1]]
                for i in range(len(reg_path) - 1)
            ]

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

    return node_transforms, None


def groupwise_resolution_global_optimization(
    g_reg,
    reference_view=None,
    transform="translation",
    max_iter=None,
    rel_tol=None,
    abs_tol=None,
    max_residual_max_mean_ratio=None,
    type_of_residual=None,
):
    """
    Get final transform parameters by global optimization.

    Output parameters P for each view map coordinates in the view
    into the coordinates of a new coordinate system.

    Strategy:
    - iterate over timepoints
    - for each pairwise registration, compute virtual pairs of beads
        - fixed view: take bounding box corners
        - moving view: transform fixed view corners using inverse of pairwise transform
    - for each view, compute the transform that minimizes the distance between
        its virtual beads and associated virtual beads in overlapping views
    - assign the computed transform to the view
    - perform this process global_optimizationly until convergence
        (transforming each view's beads by the view's current transformation)
    - potentially: filter out bad links looking at residual distances

    Data structure:
    - graph or table?
    - start with graph

    Algorithm:
    - initialise view transforms with identity transforms
    - build graph with pairwise transforms as edges
        - this is an undirected graph
    - calculate an order of views by descending connectivity / number of links
    - for each view, compute the transform that minimizes the distance between
        its virtual beads and associated virtual beads in overlapping views
    - assign the computed transform to the view

    References:
    - https://imagej.net/imagej-wiki-static/SPIM_Registration_Method
    - BigStitcher publication: https://www.nature.com/articles/s41592-019-0501-0#Sec2
      - Supplementary Note 2

    Parameters
    ----------
    g_reg : nx.Graph
        Registration graph with pairwise transforms as edges
    transform : str, optional
        Transformation type ('translation', 'rigid' or 'affine')
    max_iter : int, optional
        Maximum number of iterations
    rel_tol : float, optional
        Relative tolerance for convergence
    abs_tol : float, optional
        Absolute tolerance for convergence
    max_residual_max_mean_ratio : float, optional
        Maximum ratio of the maximum residual to the mean residual

    Returns
    -------
    dict
        Dictionary containing the final transform parameters for each view
    """

    ndim = g_reg.edges[list(g_reg.edges)[0]]["transform"].shape[-1] - 1

    # find timepoints
    all_transforms = [g_reg.edges[e]["transform"] for e in g_reg.edges]
    t_coords = np.unique(
        [
            transform.coords["t"].data
            for transform in all_transforms
            if "t" in transform.coords
        ]
    )

    params = {nodes: [] for nodes in g_reg.nodes}
    all_dfs = []
    ccs = list(nx.connected_components(g_reg))
    for icc, cc in enumerate(ccs):
        g_reg_subgraph = g_reg.subgraph(list(cc))

        if reference_view is not None and reference_view in cc:
            ref_node = reference_view
        else:
            ref_node = (
                mv_graph.get_node_with_maximal_edge_weight_sum_from_graph(
                    g_reg_subgraph, weight_key="quality"
                )
            )

        if len(t_coords):
            g_reg_subgraph_ts = [
                get_reg_graph_with_single_tp_transforms(g_reg_subgraph, t)
                for t in t_coords
            ]
        else:
            g_reg_subgraph_ts = [g_reg_subgraph]

        g_beads_subgraph_ts = [
            get_beads_graph_from_reg_graph(g_reg_subgraph_t, ndim=ndim)
            for g_reg_subgraph_t in g_reg_subgraph_ts
        ]

        cc_params, cc_dfs = list(
            zip(
                *tuple(
                    [
                        optimize_bead_subgraph(
                            g_beads_subgraph,
                            transform,
                            ref_node,
                            max_iter,
                            rel_tol,
                            abs_tol,
                            max_residual_max_mean_ratio,
                            type_of_residual,
                        )
                        for g_beads_subgraph in g_beads_subgraph_ts
                    ]
                )
            )
        )

        for node in cc:
            for cc_param in cc_params:
                params[node].append(cc_param[node])

        if len(t_coords):
            for it, t in enumerate(t_coords):
                if cc_dfs[it] is not None:
                    cc_dfs[it]["t"] = [t] * len(cc_dfs[it])
                    cc_dfs[it]["icc"] = [icc] * len(cc_dfs[it])
                    all_dfs.append(cc_dfs[it])

    all_dfs = [df for df in all_dfs if df is not None]
    df = pd.concat(all_dfs) if len(all_dfs) else None

    for node in g_reg.nodes:
        params[node] = xr.concat(params[node], dim="t").assign_coords(
            {"t": t_coords}
        )

    return params, df


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
    max_residual_max_mean_ratio,
    type_of_residual,
):
    """
    Optimize the virtual bead graph.

    Two loops:
    - first loop: iterate until max / mean residual ratio is below threshold
    - second loop: optimise transformations of each node

    TODO: improve efficiency

    Parameters
    ----------
    g_beads_subgraph : nx.Graph
        Virtual bead graph
    transform : str
        Transformation type ('rigid' or 'affine')
    ref_node : int
    max_iter : int, optional
    rel_tol : float, optional
    abs_tol : float, optional
    max_residual_max_mean_ratio : float, optional
    type_of_residual : str, optional
        Whether to calculate residuals on 'edge' or 'node' level

    Returns
    -------
    nx.Graph
        Optimized virtual bead graph
    """

    g_beads_subgraph = copy.deepcopy(g_beads_subgraph)

    if max_iter is None:
        max_iter = 100
    if rel_tol is None:
        rel_tol = 1e-5
    if abs_tol is None:
        abs_tol = 1e-7
    if max_residual_max_mean_ratio is None:
        max_residual_max_mean_ratio = 3.0
    if type_of_residual is None:
        type_of_residual = "edge"

    # calculate an order of views by descending connectivity / number of links
    centralities = nx.degree_centrality(g_beads_subgraph)
    sorted_nodes = sorted(centralities, key=centralities.get, reverse=True)

    ndim = (
        g_beads_subgraph.nodes[list(g_beads_subgraph.nodes)[0]][
            "affine"
        ].shape[-1]
        - 1
    )

    if transform == "translation":
        transform_generator = TranslationTransform(dimensionality=ndim)
    elif transform == "rigid":
        transform_generator = EuclideanTransform(dimensionality=ndim)
    elif transform == "affine":
        transform_generator = AffineTransform(dimensionality=ndim)

    mean_residuals = []
    dfs = []

    pts_per_edge = ndim**2

    total_iterations = 0
    # first loop: iterate until max / mean residual ratio is below threshold
    while True:
        # second loop: optimise transformations of each node
        edge_residuals = {tuple(sorted(e)): [] for e in g_beads_subgraph.edges}
        for iteration in range(max_iter):
            iter_residuals = []
            if not len(list(g_beads_subgraph.edges)):
                break
            new_affines = []
            for curr_node in sorted_nodes:
                node_edges = list(g_beads_subgraph.edges(curr_node))

                if len(node_edges) == 0:
                    new_affines.append(param_utils.identity_transform(ndim))
                    continue

                node_pts = transformation.transform_pts(
                    np.concatenate(
                        [
                            g_beads_subgraph.edges[e]["beads"][curr_node]
                            for ie, e in enumerate(node_edges)
                        ],
                        axis=0,
                    ),
                    g_beads_subgraph.nodes[curr_node]["affine"],
                )
                adjecent_pts = np.concatenate(
                    [
                        transformation.transform_pts(
                            g_beads_subgraph.edges[e]["beads"][n],
                            g_beads_subgraph.nodes[n]["affine"],
                        )
                        for ie, e in enumerate(node_edges)
                        for n in e
                        if n != curr_node
                    ],
                    axis=0,
                )

                ### repeat points based on edge quality
                ### (not used currently, as although it seems to improve convergence, it slows down performance)

                # edge_qualities = np.array([g_beads_subgraph.edges[e]['quality'] for e in node_edges])
                # edge_qualities_norm = edge_qualities / np.sum(edge_qualities)
                # edge_repeats = [np.max([1, int(edge_qualities_norm[ie] * ndim ** 2 * 4)])
                #     for ie in range(len(edge_qualities))]

                # pts_per_edge = ndim ** 2
                # node_pts_reg = np.concatenate(
                #     [
                #         np.repeat(
                #             node_pts[ie * pts_per_edge : (ie + 1) * pts_per_edge],
                #             edge_repeats[ie],
                #             axis=0,
                #         )
                #         for ie in range(len(node_edges))
                #     ], axis=0
                # )

                # adjecent_pts_reg = np.concatenate(
                #     [
                #         np.repeat(
                #             adjecent_pts[ie * pts_per_edge : (ie + 1) * pts_per_edge],
                #             edge_repeats[ie],
                #             axis=0,
                #         )
                #         for ie in range(len(node_edges))
                #     ], axis=0
                # )

                # transform_generator.estimate(node_pts_reg, adjecent_pts_reg)

                transform_generator.estimate(node_pts, adjecent_pts)

                curr_residuals = transform_generator.residuals(
                    node_pts, adjecent_pts
                )

                curr_edge_residuals = np.array(
                    [
                        np.mean(
                            curr_residuals[
                                ie * pts_per_edge : (ie + 1) * pts_per_edge
                            ]
                        )
                        for ie, e in enumerate(node_edges)
                    ]
                )

                for ie, e in enumerate(node_edges):
                    edge_residuals[tuple(sorted(e))].append(
                        curr_edge_residuals[ie]
                    )

                iter_residuals.append(np.mean(curr_residuals))

                if curr_node != ref_node:
                    new_affines.append(
                        param_utils.matmul_xparams(
                            param_utils.affine_to_xaffine(
                                transform_generator.params
                            ),
                            g_beads_subgraph.nodes[curr_node]["affine"],
                        )
                    )
                else:
                    new_affines.append(param_utils.identity_transform(ndim))

                total_iterations += 1

                df_iter_node = pd.concat(
                    [
                        pd.DataFrame(
                            {
                                "residual": np.mean(curr_residuals),
                                "iteration": iteration,
                                "total_iteration": total_iterations,
                            },
                            index=[str(curr_node)],
                        ),
                    ],
                    axis=1,
                ).reset_index(names=["node"])

                dfs.append(df_iter_node)

            mean_residuals.append(np.mean(iter_residuals))

            for i, curr_node in enumerate(sorted_nodes):
                g_beads_subgraph.nodes[curr_node]["affine"] = new_affines[i]

            # check for convergence
            if iteration > 5:
                abs_change = np.abs(mean_residuals[-1] - mean_residuals[-2])
                if abs_change < abs_tol:
                    break

                rel_change = (
                    np.abs(mean_residuals[-1] - mean_residuals[-2])
                    / mean_residuals[-1]
                )
                if rel_change < rel_tol:
                    break

            # print(iteration, mean_residuals[-1], iter_residuals)

        # keep parameters after one iteration if there are
        # less than two edges
        if len(list(g_beads_subgraph.edges)) < 2:
            break

        if type_of_residual == "edge":
            edge_residuals = {k: np.mean(v) for k, v in edge_residuals.items()}

            edges, edge_residual_values = zip(*edge_residuals.items())

            mean_edge_residual_value = np.mean(edge_residual_values)

            if (
                mean_edge_residual_value < abs_tol
                or np.max(edge_residual_values) / mean_edge_residual_value
                < max_residual_max_mean_ratio
            ):
                edge_to_remove = None
            else:
                edge_to_remove = edges[np.argmax(edge_residual_values)]

        elif type_of_residual == "node":
            mean_iter_residual = np.mean(iter_residuals)

            if (
                mean_iter_residual > abs_tol
                and np.max(iter_residuals) / mean_iter_residual
                > max_residual_max_mean_ratio
            ):
                # determine which edge to remove (formula from bigstitcher)
                # c_value = (1-qij)**2 * sqrt(dijk) * log10(max(degi, degj))
                c_value_dict = {
                    e: (1 / float(g_beads_subgraph.edges[e]["overlap"])) ** 2
                    * (1 - float(g_beads_subgraph.edges[e]["quality"])) ** 2
                    * np.sqrt(np.max(edge_residuals[e]))
                    * np.log10(
                        np.max(
                            [
                                len(list(g_beads_subgraph.neighbors(n)))
                                for n in e
                            ]
                        )
                    )
                    for e in g_beads_subgraph.edges
                }

                edges, c_values = zip(*c_value_dict.items())
                edge_to_remove = edges[np.argmax(c_values)]

            else:
                edge_to_remove = None
        else:
            raise ValueError(f"Unknown type of residual: {type_of_residual}")

        logger.debug("Glob opt iter %s", iteration)
        logger.debug(
            "Max and mean residuals: %s \t %s",
            {np.max(iter_residuals)},
            np.mean(iter_residuals),
        )

        if edge_to_remove is not None:
            g_beads_subgraph.remove_edge(*edge_to_remove)

            # reset view transforms with identity transforms
            for node in g_beads_subgraph.nodes:
                g_beads_subgraph.nodes[node][
                    "affine"
                ] = param_utils.identity_transform(ndim)

            logger.debug(
                "Removing edge %s and restarting glob opt.", edge_to_remove
            )
        else:
            logger.debug(
                "Finished glob opt. Max and mean residuals: %s \t %s",
                np.max(iter_residuals),
                np.mean(iter_residuals),
            )
            break

    df = pd.concat(dfs, axis=0) if len(dfs) else None

    params = {
        node: g_beads_subgraph.nodes[node]["affine"]
        for node in g_beads_subgraph.nodes
    }
    return params, df


def register(
    msims: list[MultiscaleSpatialImage],
    transform_key,
    reg_channel_index=None,
    new_transform_key=None,
    registration_binning=None,
    overlap_tolerance=0.0,
    pairwise_reg_func=phase_correlation_registration,
    pairwise_reg_func_kwargs=None,
    groupwise_resolution_method="global_optimization",
    groupwise_resolution_kwargs=None,
    pre_registration_pruning_method="alternating_pattern",
    post_registration_do_quality_filter=False,
    post_registration_quality_threshold=0.2,
    plot_summary=False,
    pairs=None,
    scheduler=None,
):
    """

    Register a list of views to a common extrinsic coordinate system.

    This function is the main entry point for registration.

    1) Build a graph of pairwise overlaps between views
    2) Determine registration pairs from this graph
    3) Register each pair of views.
       Need to add option to pass registration functions here.
    4) Determine the parameters mapping each view into the new extrinsic
       coordinate system.
       Currently done by determining a reference view and concatenating for reach
       view the pairwise transforms along the shortest paths towards the ref view.

    Parameters
    ----------
    msims : list of MultiscaleSpatialImage
        Input views
    reg_channel_index : int, optional
        Index of channel to be used for registration, by default None
    transform_key : str, optional
        Extrinsic coordinate system to use as a starting point
        for the registration, by default None
    new_transform_key : str, optional
        If set, the registration result will be registered as a new extrinsic
        coordinate system in the input views (with the given name), by default None
    registration_binning : dict, optional
        Binning applied to each dimensionn during registration, by default None
    overlap_tolerance : float, optional
        Extend overlap regions considered for pairwise registration.
        - if 0, the overlap region is the intersection of the bounding boxes.
        - if > 0, the overlap region is the intersection of the bounding boxes
            extended by this value in all spatial dimensions.
        - if None, the full images are used for registration
    pairwise_reg_func : func, optional
        Function used for registration.
    pairwise_reg_func_kwargs : dict, optional
        Additional keyword arguments passed to the registration function
    groupwise_resolution_method : str, optional
        Method used to determine the final transform parameters
        from pairwise registrations:
        - 'global_optimization': global optimization considering all pairwise transforms
        - 'shortest_paths': concatenation of pairwise transforms along shortest paths
    groupwise_resolution_kwargs : dict, optional
        Additional keyword arguments passed to the groupwise optimization function
    pre_registration_pruning_method : str, optional
        Method used to eliminate registration edges (e.g. diagonals) from the view adjacency
        graph before registration. Available methods:
        - None: No pruning, useful when no regular arrangement is present.
        - 'alternating_pattern': Prune to edges between squares of differering
            colors in checkerboard pattern. Useful for regular 2D tile arrangements (of both 2D or 3D data).
        - 'shortest_paths_overlap_weighted': Prune to shortest paths in overlap graph
            (weighted by overlap). Useful to minimize the number of pairwise registrations.
        - 'otsu_threshold_on_overlap': Prune to edges with overlap above Otsu threshold.
            This is useful for regular 2D or 3D grid arrangements, as diagonal edges will be pruned.
    post_registration_do_quality_filter : bool, optional
    post_registration_quality_threshold : float, optional
        Threshold used to filter edges by quality after registration,
        by default None (no filtering)
    plot_summary : bool, optional
        If True, plot a graph showing registered stack boundaries and
        performed pairwise registrations including correlations, by default False
    pairs : list of tuples, optional
        If set, initialises the view adjacency graph using the indicates
        pairs of view/tile indices, by default None

    Returns
    -------
    list of xr.DataArray
        Parameters mapping each view into a new extrinsic coordinate system
    """

    if pairwise_reg_func_kwargs is None:
        pairwise_reg_func_kwargs = {}

    if groupwise_resolution_kwargs is None:
        groupwise_resolution_kwargs = {}

    sims = [msi_utils.get_sim_from_msim(msim) for msim in msims]

    if reg_channel_index is None:
        for msim in msims:
            if "c" in msi_utils.get_dims(msim):
                raise (Exception("Please choose a registration channel."))

    msims_reg = [
        msi_utils.multiscale_sel_coords(
            msim, {"c": sims[imsim].coords["c"][reg_channel_index]}
        )
        if "c" in msi_utils.get_dims(msim)
        else msim
        for imsim, msim in enumerate(msims)
    ]

    g = mv_graph.build_view_adjacency_graph_from_msims(
        msims_reg,
        transform_key=transform_key,
        pairs=pairs,
    )

    if pre_registration_pruning_method is not None:
        g_reg = prune_view_adjacency_graph(
            g,
            method=pre_registration_pruning_method,
        )
    else:
        g_reg = g

    g_reg_computed = compute_pairwise_registrations(
        msims_reg,
        g_reg,
        transform_key=transform_key,
        registration_binning=registration_binning,
        overlap_tolerance=overlap_tolerance,
        pairwise_reg_func=pairwise_reg_func,
        pairwise_reg_func_kwargs=pairwise_reg_func_kwargs,
        scheduler=scheduler,
    )

    if post_registration_do_quality_filter:
        # filter edges by quality
        g_reg_computed = mv_graph.filter_edges(
            g_reg_computed,
            threshold=post_registration_quality_threshold,
            weight_key="quality",
        )

    params, groupwise_opt_info = groupwise_resolution(
        g_reg_computed,
        method=groupwise_resolution_method,
        **groupwise_resolution_kwargs,
    )

    params = [params[iview] for iview in sorted(g_reg_computed.nodes())]

    if new_transform_key is not None:
        for imsim, msim in enumerate(msims):
            msi_utils.set_affine_transform(
                msim,
                params[imsim],
                transform_key=new_transform_key,
                base_transform_key=transform_key,
            )

        if plot_summary:
            edges = list(g_reg_computed.edges())
            _fig, _ax = vis_utils.plot_positions(
                msims,
                transform_key=new_transform_key,
                edges=edges,
                edge_color_vals=np.array(
                    [
                        g_reg_computed.get_edge_data(*e)["quality"].mean()
                        for e in edges
                    ]
                ),
                edge_label="pairwise view correlation",
                display_view_indices=True,
                use_positional_colors=False,
            )

    return params  # , groupwise_opt_info


def compute_pairwise_registrations(
    msims,
    g_reg,
    transform_key=None,
    registration_binning=None,
    overlap_tolerance=0.0,
    pairwise_reg_func=phase_correlation_registration,
    pairwise_reg_func_kwargs=None,
    scheduler=None,
):
    g_reg_computed = g_reg.copy()
    edges = [tuple(sorted([e[0], e[1]])) for e in g_reg.edges]

    params_xds = [
        register_pair_of_msims_over_time(
            msims[pair[0]],
            msims[pair[1]],
            transform_key=transform_key,
            registration_binning=registration_binning,
            overlap_tolerance=overlap_tolerance,
            pairwise_reg_func=pairwise_reg_func,
            pairwise_reg_func_kwargs=pairwise_reg_func_kwargs,
        )
        for pair in edges
    ]

    params = compute(params_xds, scheduler=scheduler)[0]

    for i, pair in enumerate(edges):
        g_reg_computed.edges[pair]["transform"] = params[i]["transform"]
        g_reg_computed.edges[pair]["quality"] = params[i]["quality"]
        g_reg_computed.edges[pair]["bbox"] = params[i]["bbox"]

    return g_reg_computed


def crop_sim_to_references(
    sim_input_to_crop,
    reference_sims,
    transform_key_input,
    transform_keys_reference,
    input_time_index=0,
):
    """
    Crop input image to the minimal region fully covering the reference sim(s).
    """

    ref_corners_world = []
    for irefsim, reference_sim in enumerate(reference_sims):
        ref_corners_world += list(
            np.unique(
                mv_graph.get_vertices_from_stack_props(
                    spatial_image_utils.get_stack_properties_from_sim(
                        reference_sim,
                        transform_key=transform_keys_reference[irefsim],
                    )
                ).reshape((-1, 2)),
                axis=0,
            )
        )

    input_affine = spatial_image_utils.get_affine_from_sim(
        sim_input_to_crop, transform_key=transform_key_input
    )

    if "t" in input_affine.dims:
        input_affine = input_affine.sel(
            {"t": input_affine.coords["t"][input_time_index]}
        )

    input_affine_inv = np.linalg.inv(np.array(input_affine))
    ref_corners_input_dataspace = transformation.transform_pts(
        ref_corners_world, np.array(input_affine_inv)
    )

    lower, upper = (
        func(ref_corners_input_dataspace, axis=0) for func in [np.min, np.max]
    )

    sdims = spatial_image_utils.get_spatial_dims_from_sim(sim_input_to_crop)

    sim_cropped = spatial_image_utils.sim_sel_coords(
        sim_input_to_crop,
        {
            dim: (sim_input_to_crop.coords[dim] > lower[idim])
            * (sim_input_to_crop.coords[dim] < upper[idim])
            for idim, dim in enumerate(sdims)
        },
    )

    return sim_cropped


def registration_ANTsPy(
    fixed_data,
    moving_data,
    *,
    fixed_origin,
    moving_origin,
    fixed_spacing,
    moving_spacing,
    initial_affine,
    transform_types=None,
    **ants_registration_kwargs,
):
    """
    Use ANTsPy to perform registration between two spatial images.
    """

    if ants is None:
        raise (
            Exception(
                """
Please install the antspyx package to use ANTsPy for registration.
E.g. using pip:
- `pip install multiview-stitcher[ants]` or
- `pip install antspyx`
"""
            )
        )

    if transform_types is None:
        transform_types = ["Translation", "Rigid", "Similarity"]

    spatial_dims = fixed_data.dims
    ndim = len(fixed_spacing)

    # convert input images to ants images
    fixed_ants = ants.from_numpy(
        fixed_data.astype(np.float32),
        origin=[fixed_origin[dim] for dim in spatial_dims][::-1],
        spacing=[fixed_spacing[dim] for dim in spatial_dims][::-1],
    )
    moving_ants = ants.from_numpy(
        moving_data.astype(np.float32),
        origin=[moving_origin[dim] for dim in spatial_dims][::-1],
        spacing=[moving_spacing[dim] for dim in spatial_dims][::-1],
    )

    init_aff = ants.ants_transform_io.create_ants_transform(
        transform_type="AffineTransform",
        dimension=ndim,
        matrix=np.array(initial_affine)[:ndim, :ndim][::-1, ::-1],
        offset=np.array(initial_affine)[:ndim, ndim][::-1],
    )

    default_ants_registration_kwargs = {
        "random_seed": 0,
        "write_composite_transform": False,
        "aff_metric": "mattes",
        # "aff_metric": "meansquares",
        "verbose": False,
        "aff_random_sampling_rate": 0.2,
        # "aff_iterations": (2000, 2000, 1000, 100),
        # "aff_smoothing_sigmas": (4, 2, 1, 0),
        # "aff_shrink_factors": (6, 4, 2, 1),
        "aff_iterations": (2000, 2000),
        "aff_smoothing_sigmas": (1, 0),
        "aff_shrink_factors": (2, 1),
    }

    ants_registration_kwargs = {
        **default_ants_registration_kwargs,
        **ants_registration_kwargs,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        init_transform_path = os.path.join(tmpdir, "init_aff.txt")
        ants.ants_transform_io.write_transform(init_aff, init_transform_path)

        for transform_type in transform_types:
            aff = ants.registration(
                fixed=fixed_ants,
                moving=moving_ants,
                type_of_transform=transform_type,
                initial_transform=[init_transform_path],
                **ants_registration_kwargs,
            )

            # ants.registration(fixed, moving, type_of_transform='SyN', initial_transform=None, outprefix='', mask=None, moving_mask=None, mask_all_stages=False,
            # grad_step=0.2, flow_sigma=3, total_sigma=0, aff_metric='mattes', aff_sampling=32, aff_random_sampling_rate=0.2,
            # syn_metric='mattes', syn_sampling=32, reg_iterations=(40, 20, 0),
            # aff_iterations=(2100, 1200, 1200, 10), aff_shrink_factors=(6, 4, 2, 1), aff_smoothing_sigmas=(3, 2, 1, 0),
            # write_composite_transform=False, random_seed=None, verbose=False, multivariate_extras=None, restrict_transformation=None, smoothing_in_mm=False, **kwargs)

            result_transform_path = aff["fwdtransforms"][0]
            result_transform = ants.ants_transform_io.read_transform(
                result_transform_path
            )

            curr_init_transform = result_transform
            ants.ants_transform_io.write_transform(
                curr_init_transform, init_transform_path
            )

    # reverse engineer the affine matrix from ants output parameters
    # linearising is not enough as there seems to be a centering convention affecting the translation part
    gv = np.array(list(np.ndindex(tuple([2] * ndim))))
    gv_t = np.array([result_transform.apply_to_point(pt) for pt in gv])
    simage_affine = AffineTransform()
    simage_affine.estimate(gv, gv_t)
    p = simage_affine.params

    # ants coordinates are in xyz order
    p = param_utils.invert_coordinate_order(p)

    p = param_utils.affine_to_xaffine(p)

    quality = link_quality_metric_func(
        fixed_ants.numpy(), aff["warpedmovout"].numpy()
    )

    reg_result = {}

    reg_result["affine_matrix"] = p
    reg_result["quality"] = quality

    return reg_result
