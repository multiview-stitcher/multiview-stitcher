import copy
import logging
import os
import tempfile
import warnings
from typing import Union

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
from skimage.exposure import rescale_intensity
from skimage.metrics import structural_similarity
from skimage.transform import (
    EuclideanTransform,
    SimilarityTransform,
)

from multiview_stitcher.transforms import AffineTransform, TranslationTransform

try:
    import ants
except ImportError:
    ants = None

from multiview_stitcher import (
    fusion,
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
    quality = stats.spearmanr(im0.flatten(), im1t.flatten()).correlation
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

    Implementation:
      - input are two spatial images which already overlap
      - start with binning of 1
      - if total number of pixels in the stack is too large, double binning of the dimension
        with smallest spacing (with x and y tied to each other)
    """

    spatial_dims = spatial_image_utils.get_spatial_dims_from_sim(sim1)
    ndim = len(spatial_dims)
    input_spacings = [
        spatial_image_utils.get_spacing_from_sim(sim, asarray=False)
        for sim in [sim1, sim2]
    ]

    if overlap_tolerance is not None:
        raise (NotImplementedError("overlap_tolerance"))

    overlap = {
        dim: max(sim1.shape[idim], sim2.shape[idim])
        for idim, dim in enumerate(spatial_dims)
    }

    registration_binning = {dim: 1 for dim in spatial_dims}
    spacings = input_spacings
    while (
        max(
            [
                np.prod(
                    [
                        overlap[dim] / registration_binning[dim]
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
            registration_binning["z"] = registration_binning["z"] + 1
        else:
            for dim in ["x", "y"]:
                registration_binning[dim] = registration_binning[dim] + 1

        spacings = [
            {
                dim: input_spacings[isim][dim] * registration_binning[dim]
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
    overlap_tolerance=None,
):
    """
    Get bounding box(es) of overlap between two spatial images
    in coord system given by input_transform_key, transformed
    into coord system given by output_transform_key (intrinsic
    coordinates if None).

    Return: lower and upper bounds of overlap for both input images
    """

    ndim = spatial_image_utils.get_ndim_from_sim(sim1)

    stack_propss = [
        spatial_image_utils.get_stack_properties_from_sim(
            sim, transform_key=input_transform_key
        )
        for sim in [sim1, sim2]
    ]

    if overlap_tolerance is not None:
        stack_propss = [
            spatial_image_utils.extend_stack_props(
                stack_props, extend_by=overlap_tolerance
            )
            for stack_props in stack_propss
        ]

    corners = [
        mv_graph.get_vertices_from_stack_props(stack_props).reshape(-1, ndim)
        for stack_props in stack_propss
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
        for isim in range(2)
    ]
    uppers = [
        np.min(np.max(corners_target_space[isim], axis=1), axis=0)
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
        np.uint64
    )

    reg_sims_b_t = [
        transformation.transform_sim(
            sim.astype(np.float32),
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
            mode="constant",
            cval=np.nan,
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


def phase_correlation_registration(
    fixed_data,
    moving_data,
    disambiguate_region_mode=None,
    **skimage_phase_corr_kwargs,
):
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

    # normalize images
    im0, im1 = (
        rescale_intensity(
            im,
            in_range=(np.nanmin(im), np.nanmax(im)),
            out_range=(0, 1),
        )
        for im in [im0, im1]
    )

    im0nm = np.isnan(im0)
    im1nm = np.isnan(im1)

    # use intersection mode if there are nan pixels in either image
    if disambiguate_region_mode is None:
        if np.any([im0nm, im1nm]):
            disambiguate_region_mode = "intersection"
        else:
            disambiguate_region_mode = "union"

    valid_pixels1 = np.sum(~im1nm)

    if np.any([im0nm, im1nm]):
        im0nn = np.nan_to_num(im0)
        im1nn = np.nan_to_num(im1)
    else:
        im0nn = im0
        im1nn = im1

    if "upsample_factor" not in skimage_phase_corr_kwargs:
        skimage_phase_corr_kwargs["upsample_factor"] = 10 if ndim == 2 else 2

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
                    im0nn,
                    im1nn,
                    disambiguate=False,
                    normalization=normalization,
                    **skimage_phase_corr_kwargs,
                )[0]
            )

        if np.any([im0nm, im1nm]):
            shift_candidates.append(
                skimage.registration.phase_cross_correlation(
                    im0,
                    im1,
                    reference_mask=im0nm,
                    moving_mask=im1nm,
                    disambiguate=False,
                    **skimage_phase_corr_kwargs,
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

    data_range = np.nanmax([im0, im1]) - np.nanmin([im0, im1])
    im1_min = np.nanmin(im1)

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

    def get_bb_from_nanmask(mask):
        bbs = []
        for idim in range(mask.ndim):
            axes = list(range(mask.ndim))
            axes.remove(idim)
            valids = np.where(np.max(mask, axis=tuple(axes)))
            bbs.append([np.min(valids), np.max(valids)])
        return bbs

    im0_bb = get_bb_from_nanmask(~im0nm)

    for t_ in t_candidates:
        im1t = ndimage.affine_transform(
            im1,
            param_utils.affine_from_translation(list(t_)),
            order=1,
            mode="constant",
            cval=np.nan,
        )
        mask = ~np.isnan(im1t) * ~im0nm

        if np.all(~mask) or float(np.sum(mask)) / valid_pixels1 < 0.1:
            disambiguate_metric_val = -1
            quality_metric_val = -1
        else:
            im1t_bb = get_bb_from_nanmask(~np.isnan(im1t))

            if disambiguate_region_mode == "union":
                mask_slices = tuple(
                    [
                        slice(
                            min(im0_bb[idim][0], im1t_bb[idim][0]),
                            max(im0_bb[idim][1], im1t_bb[idim][1]) + 1,
                        )
                        for idim in range(ndim)
                    ]
                )
            elif disambiguate_region_mode == "intersection":
                mask_slices = tuple(
                    [
                        slice(
                            max(im0_bb[idim][0], im1t_bb[idim][0]),
                            min(im0_bb[idim][1], im1t_bb[idim][1]) + 1,
                        )
                        for idim in range(ndim)
                    ]
                )

            if np.nanmax(im1t[mask_slices]) <= im1_min:
                disambiguate_metric_val = -1
                quality_metric_val = -1
                continue

            # structural similarity seems to be better than
            # correlation for disambiguation (need to solidify this)
            min_shape = np.min(im0[mask_slices].shape)
            ssim_win_size = np.min([7, min_shape - ((min_shape - 1) % 2)])
            if ssim_win_size < 3 or np.max(im1t[mask_slices]) <= im1_min:
                logger.debug("SSIM window size too small")
                disambiguate_metric_val = -1
            else:
                disambiguate_metric_val = structural_similarity(
                    np.nan_to_num(im0[mask_slices]),
                    np.nan_to_num(im1t[mask_slices]),
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


def dispatch_pairwise_reg_func(
    pairwise_reg_func, fixed_data, moving_data, **pairwise_reg_func_kwargs
):
    """
    Check that images are not constant and apply the registration function.
    """
    int_extrema = [
        [func(im) for im in [fixed_data, moving_data]]
        for func in [np.nanmin, np.nanmax]
    ]

    # return if no translation if images are constant
    for i in range(2):
        if int_extrema[0][i] == int_extrema[1][i]:
            warnings.warn(
                "An overlap region between tiles/views is all zero or constant. Assuming identity transform.",
                UserWarning,
                stacklevel=2,
            )
            reg_result = {}
            reg_result["affine_matrix"] = param_utils.identity_transform(
                fixed_data.ndim
            )
            reg_result["quality"] = np.nan
            return reg_result

    return pairwise_reg_func(
        fixed_data, moving_data, **pairwise_reg_func_kwargs
    )


def register_pair_of_msims(
    msim1,
    msim2,
    transform_key,
    registration_binning=None,
    overlap_tolerance: Union[int, dict[str, int]] = None,
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
    pairwise_reg_func : Callable, optional
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

    if overlap_tolerance is None:
        overlap_tolerance = {dim: 0.0 for dim in spatial_dims}
    elif isinstance(overlap_tolerance, (int, float)):
        overlap_tolerance = {
            dim: float(overlap_tolerance) for dim in spatial_dims
        }
    elif isinstance(overlap_tolerance, dict):
        overlap_tolerance = {
            dim: float(overlap_tolerance[dim])
            if dim in overlap_tolerance
            else 0.0
            for dim in spatial_dims
        }

    sim1 = msi_utils.get_sim_from_msim(msim1)
    sim2 = msi_utils.get_sim_from_msim(msim2)

    reg_sims = [sim1, sim2]

    if registration_binning is None:
        logger.info("Determining optimal registration binning")
        registration_binning = get_optimal_registration_binning(
            reg_sims[0], reg_sims[1]
        )

    # logging without use of %s
    logger.info("Registration binning: %s", registration_binning)

    if max(registration_binning.values()) > 1:
        reg_sims_b = [
            sim.coarsen(registration_binning, boundary="trim")
            .mean()
            .astype(sim.dtype)
            for sim in reg_sims
        ]
    else:
        reg_sims_b = reg_sims

    lowers, uppers = get_overlap_bboxes(
        reg_sims_b[0],
        reg_sims_b[1],
        input_transform_key=transform_key,
        output_transform_key=None,
        overlap_tolerance=overlap_tolerance,
    )

    reg_sims_spacing = [
        spatial_image_utils.get_spacing_from_sim(sim) for sim in reg_sims_b
    ]

    tol = 1e-6
    reg_sims_b = [
        sim.sel(
            {
                # add spacing to include bounding pixels
                dim: slice(
                    lowers[isim][idim] - tol - reg_sims_spacing[isim][dim],
                    uppers[isim][idim] + tol + reg_sims_spacing[isim][dim],
                )
                for idim, dim in enumerate(spatial_dims)
            },
        )
        for isim, sim in enumerate(reg_sims_b)
    ]

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

    param_dict_d = delayed(dispatch_pairwise_reg_func, nout=1)(
        pairwise_reg_func,
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
        overlap_tolerance=overlap_tolerance,
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
    **register_kwargs,
):
    """
    Apply register_pair_of_msims to each time point of the input images.
    """

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
                    **register_kwargs,
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
    - 'keep_axis_aligned':
        Keep only edges that align with the axes of the
        tiles. This is useful for regular grid arrangements,
        in which case it excludes 'diagonal' edges.
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
    elif method == "alternating_pattern":
        return mv_graph.prune_graph_to_alternating_colors(
            g, return_colors=False
        )
    elif method == "shortest_paths_overlap_weighted":
        return mv_graph.prune_to_shortest_weighted_paths(g)
    elif method == "otsu_threshold_on_overlap":
        return mv_graph.filter_edges(g)
    elif method == "keep_axis_aligned":
        return mv_graph.prune_to_axis_aligned_edges(g)
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

    ndim = (
        g_reg.get_edge_data(*list(g_reg.edges())[0])["transform"].shape[-1] - 1
    )

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
        paths = {
            n: nx.shortest_path(
                subgraph, target=n, source=ref_node, weight="quality_mean_inv"
            )
            for n in cc
        }

        for n in subgraph.nodes:
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
):
    """
    Get final transform parameters by global optimization.

    Output parameters P for each view map coordinates in the view
    into the coordinates of a new coordinate system.

    Strategy:
    - iterate over timepoints
    - iterate over connected components of the registration graph
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
    cc_g_opt_t0s = []
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

        cc_params, cc_dfs, cc_g_opt_ts = list(
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
                        )
                        for g_beads_subgraph in g_beads_subgraph_ts
                    ]
                )
            )
        )

        cc_g_opt_t0s.append(cc_g_opt_ts[0])

        for node in cc:
            for cc_param in cc_params:
                params[node].append(cc_param[node])

        if len(t_coords):
            for it, t in enumerate(t_coords):
                if cc_dfs[it] is not None:
                    cc_dfs[it]["t"] = [t] * len(cc_dfs[it])
                    cc_dfs[it]["icc"] = [icc] * len(cc_dfs[it])
                    all_dfs.append(cc_dfs[it])

    # join optimized graphs for first timepoint

    g_opt_t0 = nx.compose_all(cc_g_opt_t0s)

    all_dfs = [df for df in all_dfs if df is not None]
    df = pd.concat(all_dfs) if len(all_dfs) else None

    for node in g_reg.nodes:
        params[node] = xr.concat(params[node], dim="t").assign_coords(
            {"t": t_coords}
        )

    info_dict = {
        "metrics": df,
        "optimized_graph_t0": g_opt_t0,
    }

    return params, info_dict


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
                #         for ie in range(len(node_edges[curr_node]))
                #     ], axis=0
                # )

                # adjecent_pts_reg = np.concatenate(
                #     [
                #         np.repeat(
                #             adjecent_pts[ie * pts_per_edge : (ie + 1) * pts_per_edge],
                #             edge_repeats[ie],
                #             axis=0,
                #         )
                #         for ie in range(len(node_edges[curr_node]))
                #     ], axis=0
                # )

                if curr_node != ref_node:
                    transform_generator.estimate(node_pts, adj_pts)
                    transform_generator.residuals(node_pts, adj_pts)

                    new_affines[curr_node] = param_utils.matmul_xparams(
                        param_utils.affine_to_xaffine(
                            transform_generator.params
                        ),
                        new_affines[curr_node],
                    ).data

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


def register(
    msims: list[MultiscaleSpatialImage],
    transform_key: str = None,
    reg_channel_index: int = None,
    reg_channel: str = None,
    new_transform_key: str = None,
    registration_binning: dict[str, int] = None,
    overlap_tolerance: Union[int, dict[str, int]] = 0.0,
    pairwise_reg_func=phase_correlation_registration,
    pairwise_reg_func_kwargs: dict = None,
    groupwise_resolution_method="global_optimization",
    groupwise_resolution_kwargs: dict = None,
    pre_registration_pruning_method="alternating_pattern",
    post_registration_do_quality_filter: bool = False,
    post_registration_quality_threshold: float = 0.2,
    plot_summary: bool = False,
    pairs: list[tuple[int, int]] = None,
    scheduler=None,
    n_parallel_pairwise_regs: int = None,
    return_dict: bool = False,
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
    reg_channel : str, optional
        Name of channel to be used for registration, by default None
        Overrides reg_channel_index
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
    pairwise_reg_func : Callable, optional
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
        - 'keep_axis_aligned': Keep only edges that align with tile axes. This is useful for regular grid
            arrangements and to explicitely prune diagonals, e.g. when other methods fail.
    post_registration_do_quality_filter : bool, optional
    post_registration_quality_threshold : float, optional
        Threshold used to filter edges by quality after registration,
        by default None (no filtering)
    plot_summary : bool, optional
        If True (and `new_transform_key` is set), plot graphs summarising the registration process and results:
        1) Cross correlation values of pairwise registrations
           (stack boundaries shown as before registration)
        2) Residual distances between registration edges after global parameter resolution.
           Grey edges have been removed during glob param res (stack boundaries shown as after registration).
        Stack boundary positions reflect the registration result.
        By default False
    pairs : list of tuples, optional
        If set, initialises the view adjacency graph using the indicates
        pairs of view/tile indices, by default None
    scheduler : str, optional
        Dask scheduler to use for parallel computation, by default None
    n_parallel_pairwise_regs : int, optional
        Number of parallel pairwise registrations to run. Setting this is specifically
        useful for limiting memory usage.
        By default None (all pairwise registrations are run in parallel)
    return_dict : bool, optional
        If True, return a dict containing params, registration metrics and more, by default False

    Returns
    -------
    list of xr.DataArray
        Parameters mapping each view into a new extrinsic coordinate system
    or
    dict
        Dictionary containing the following keys:
        - 'params': Parameters mapping each view into a new extrinsic coordinate system
        - 'pairwise_registration': Dictionary containing the following
            - 'summary_plot': Tuple containing the figure and axis of the summary plot
            - 'graph': networkx graph of pairwise registrations
            - 'metrics': Dictionary containing the following metrics:
                - 'qualities': Edge registration qualities
        - 'groupwise_resolution': Dictionary containing the following
            - 'summary_plot': Tuple containing the figure and axis of the summary plot
            - 'graph': networkx graph of groupwise resolution
            - 'metrics': Dictionary containing the following metrics:
                - 'residuals': Edge residuals after groupwise resolution
    """

    if pairwise_reg_func_kwargs is None:
        pairwise_reg_func_kwargs = {}

    if groupwise_resolution_kwargs is None:
        groupwise_resolution_kwargs = {}

    sims = [msi_utils.get_sim_from_msim(msim) for msim in msims]

    if "c" in msi_utils.get_dims(msims[0]):
        if reg_channel is None:
            if reg_channel_index is None:
                for msim in msims:
                    if "c" in msi_utils.get_dims(msim):
                        raise (
                            Exception("Please choose a registration channel.")
                        )
            else:
                reg_channel = sims[0].coords["c"][reg_channel_index]

        msims_reg = [
            msi_utils.multiscale_sel_coords(msim, {"c": reg_channel})
            if "c" in msi_utils.get_dims(msim)
            else msim
            for imsim, msim in enumerate(msims)
        ]
    else:
        msims_reg = msims

    g = mv_graph.build_view_adjacency_graph_from_msims(
        msims_reg,
        transform_key=transform_key,
        pairs=pairs,
        overlap_tolerance=overlap_tolerance,
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
        n_parallel_pairwise_regs=n_parallel_pairwise_regs,
    )

    if post_registration_do_quality_filter:
        # filter edges by quality
        g_reg_computed = mv_graph.filter_edges(
            g_reg_computed,
            threshold=post_registration_quality_threshold,
            weight_key="quality",
        )

    params, groupwise_resolution_info_dict = groupwise_resolution(
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

        if plot_summary or return_dict:
            edges = list(g_reg_computed.edges())
            fig_pair_reg, ax_pair_reg = vis_utils.plot_positions(
                msims,
                transform_key=transform_key,
                edges=edges,
                edge_color_vals=np.array(
                    [
                        g_reg_computed.get_edge_data(*e)["quality"].mean()
                        for e in edges
                    ]
                ),
                edge_label="Pairwise view correlation",
                display_view_indices=True,
                use_positional_colors=False,
                plot_title="Pairwise registration summary",
                show_plot=plot_summary,
            )

            if groupwise_resolution_info_dict is not None:
                edge_residuals = np.array(
                    [
                        groupwise_resolution_info_dict[
                            "optimized_graph_t0"
                        ].get_edge_data(*e)["residual"]
                        if e
                        in groupwise_resolution_info_dict[
                            "optimized_graph_t0"
                        ].edges
                        else np.nan
                        for e in edges
                    ]
                )
                edge_clims = [
                    np.nanmin(edge_residuals),
                    np.nanmax(edge_residuals),
                ]
                if edge_clims[0] == edge_clims[1]:
                    edge_clims = [0, 1]
                fig_group_res, ax_group_res = vis_utils.plot_positions(
                    msims,
                    transform_key=new_transform_key,
                    edges=edges,
                    edge_color_vals=edge_residuals,
                    edge_cmap="Spectral_r",
                    edge_clims=edge_clims,
                    edge_label="Remaining edge residuals [distance units]",
                    display_view_indices=True,
                    use_positional_colors=False,
                    plot_title="Global parameter resolution summary",
                    show_plot=plot_summary,
                )
            else:
                fig_group_res, ax_group_res = None, None
    else:
        fig_pair_reg, ax_pair_reg, fig_group_res, ax_group_res = [
            None,
            None,
            None,
            None,
        ]

    if return_dict:
        # limit output graphs to first timepoint (for now)
        g_reg_computed = g_reg_computed.copy()
        for e in g_reg_computed.edges:
            for k, v in g_reg_computed.edges[e].items():
                if isinstance(v, xr.DataArray) and "t" in v.coords:
                    g_reg_computed.edges[e][k] = g_reg_computed.edges[e][
                        k
                    ].sel({"t": g_reg_computed.edges[e][k].coords["t"][0]})

        return {
            "params": params,
            "pairwise_registration": {
                "summary_plot": (fig_pair_reg, ax_pair_reg),
                "graph": g_reg_computed,
                "metrics": {
                    "qualities": nx.get_edge_attributes(
                        g_reg_computed, "quality"
                    ),
                },
            },
            "groupwise_resolution": {
                "summary_plot": (fig_group_res, ax_group_res),
                "graph": groupwise_resolution_info_dict["optimized_graph_t0"],
                "metrics": {
                    "residuals": nx.get_edge_attributes(
                        groupwise_resolution_info_dict["optimized_graph_t0"],
                        "residual",
                    ),
                },
            },
        }
    else:
        return params


def compute_pairwise_registrations(
    msims,
    g_reg,
    scheduler=None,
    n_parallel_pairwise_regs=None,
    **register_kwargs,
):
    g_reg_computed = g_reg.copy()
    edges = [tuple(sorted([e[0], e[1]])) for e in g_reg.edges]

    params_xds = [
        register_pair_of_msims_over_time(
            msims[pair[0]],
            msims[pair[1]],
            **register_kwargs,
        )
        for pair in edges
    ]

    # In case of 3D data, compute pairwise registrations sequentially.
    # Transformations are still computed in parallel for each pair.
    # This is to be conservative with memory usage until a more advanced
    # memory management is implemented. Ideally, registration methods
    # should report their memory usage and we can use this information
    # to annotate the dask graph.

    if n_parallel_pairwise_regs is None:
        ndim = msi_utils.get_ndim(msims[0])
        if ndim == 3:
            n_parallel_pairwise_regs = 1
            logger.info("Setting n_parallel_pairwise_regs to 1 for 3D data")

    # report on how many pairwise registrations are computed in parallel
    if n_parallel_pairwise_regs is None:
        logger.info("Computing all pairwise registrations in parallel")
        params = compute(params_xds, scheduler=scheduler)[0]
    elif n_parallel_pairwise_regs > 0:
        logger.info(
            "Pairwise registration(s) run in parallel: %s",
            n_parallel_pairwise_regs,
        )
        params = []
        for i in range(0, len(params_xds), n_parallel_pairwise_regs):
            params += compute(
                params_xds[i : i + n_parallel_pairwise_regs],
                scheduler=scheduler,
            )[0]

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

    changed_units = False
    # fixed_spacing_orig = fixed_spacing.copy()
    # moving_spacing_orig = moving_spacing.copy()
    # fixed_origin_orig = fixed_origin.copy()
    # moving_origin_orig = moving_origin.copy()

    # there's an ants problem with small spacings and mattes mutual information
    # https://github.com/ANTsX/ANTs/issues/1348
    if (
        min(
            [fixed_spacing[dim] for dim in spatial_dims]
            + [moving_spacing[dim] for dim in spatial_dims]
        )
        < 1e-3
    ):
        logger.info("Scaling units for ANTsPy registration.")
        changed_units = True
        unit_scale_factor = 1e3
        for dim in spatial_dims:
            fixed_spacing[dim] = unit_scale_factor * fixed_spacing[dim]
            moving_spacing[dim] = unit_scale_factor * moving_spacing[dim]
            fixed_origin[dim] = unit_scale_factor * fixed_origin[dim]
            moving_origin[dim] = unit_scale_factor * moving_origin[dim]

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

    if changed_units:
        p.data[:ndim, ndim] = p.data[:ndim, ndim] / unit_scale_factor

    quality = link_quality_metric_func(
        fixed_ants.numpy(), aff["warpedmovout"].numpy()
    )

    reg_result = {}

    reg_result["affine_matrix"] = p
    reg_result["quality"] = quality

    return reg_result


def get_pairs_from_sample_masks(
    mask_sims,
    transform_key="affine_manual",
    fused_mask_spacing=None,
):
    """
    Find pairs of tiles that have overlapping/touching masks.
    Masks are assumed to be binary and can e.g. represent a sample segmentation.
    """

    with xr.set_options(keep_attrs=True):
        label_sims = [
            mask_sim * (i + 1) for i, mask_sim in enumerate(mask_sims)
        ]

    if fused_mask_spacing is None:
        fused_mask_spacing = spatial_image_utils.get_spacing_from_sim(
            label_sims[0]
        )

    fused_labels = fusion.fuse(
        label_sims,
        transform_key=transform_key,
        fusion_func=lambda transformed_views: np.nanmin(
            transformed_views, axis=0
        ),
        interpolation_order=0,
        output_spacing=fused_mask_spacing,
    ).compute()

    fused_labels = fused_labels.compute()

    pairs = mv_graph.get_connected_labels(
        fused_labels.data, structure=np.ones((3, 3, 3))
    )

    return pairs, fused_labels
