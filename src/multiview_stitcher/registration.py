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

from multiview_stitcher.param_resolution import groupwise_resolution
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
    reg_res_level=None,
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
        Binning factors to apply for registration. If reg_res_level is also provided,
        the binning factors must be compatible with the resolution level.
    reg_res_level : int, optional
        Resolution level to use for registration (e.g., 0 for scale0, 1 for scale1).
        If None, resolution level is automatically determined from registration_binning.
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

    # Determine which resolution level to use
    if reg_res_level is not None and registration_binning is not None:
        # Both specified: validate compatibility
        scale_key = f"scale{reg_res_level}"
        # Validate that the scale exists
        if scale_key not in msi_utils.get_sorted_scale_keys(msim1):
            raise ValueError(
                f"Resolution level {reg_res_level} (scale{reg_res_level}) "
                f"does not exist in the multiscale image"
            )
        
        # Get sims at the specified resolution level
        sim1 = msi_utils.get_sim_from_msim(msim1, scale=scale_key)
        sim2 = msi_utils.get_sim_from_msim(msim2, scale=scale_key)
        
        # Check if binning is compatible with this resolution level
        sim0_1 = msi_utils.get_sim_from_msim(msim1, scale="scale0")
        shape0 = {dim: len(sim0_1.coords[dim]) for dim in spatial_dims}
        shape_level = {dim: len(sim1.coords[dim]) for dim in spatial_dims}
        
        actual_factors = {
            dim: shape0[dim] / shape_level[dim] for dim in spatial_dims
        }
        
        for dim in spatial_dims:
            if dim in registration_binning:
                actual_factor = int(round(actual_factors[dim]))
                if registration_binning[dim] % actual_factor != 0:
                    raise ValueError(
                        f"Resolution level {reg_res_level} has downsampling "
                        f"factor {actual_factor} for dimension {dim}, which is "
                        f"not a divisor of registration_binning[{dim}]="
                        f"{registration_binning[dim]}"
                    )
                
        # calculate remaining binning to be applied after selecting the resolution level
        registration_binning = {
            dim: registration_binning[dim] // int(round(actual_factors[dim]))
            for dim in spatial_dims
        }
        
        # registration_binning will be applied below
        
    elif reg_res_level is not None:
        # Only reg_res_level specified: use that level directly
        scale_key = f"scale{reg_res_level}"
        if scale_key not in msi_utils.get_sorted_scale_keys(msim1):
            raise ValueError(
                f"Resolution level {reg_res_level} (scale{reg_res_level}) "
                f"does not exist in the multiscale image"
            )
        sim1 = msi_utils.get_sim_from_msim(msim1, scale=scale_key)
        sim2 = msi_utils.get_sim_from_msim(msim2, scale=scale_key)
        registration_binning = {dim: 1 for dim in spatial_dims}

    else:
        if registration_binning is None:
            logger.info("Determining optimal registration binning")
            sim1_0 = msi_utils.get_sim_from_msim(msim1, scale="scale0")
            sim2_0 = msi_utils.get_sim_from_msim(msim2, scale="scale0")
            registration_binning = get_optimal_registration_binning(
                sim1_0, sim2_0
            )
            logger.info("Determined optimal registration binning to be %s",
                registration_binning)

        # Only registration_binning specified: find optimal resolution level
        scale_key, remaining_binning = msi_utils.get_res_level_from_binning_factors(
            msim1, registration_binning
        )
        logger.info(
            "Using resolution level %s for registration with remaining binning %s",
            scale_key, remaining_binning
        )
        sim1 = msi_utils.get_sim_from_msim(msim1, scale=scale_key)
        sim2 = msi_utils.get_sim_from_msim(msim2, scale=scale_key)
        registration_binning = remaining_binning
        logger.info('Determined reg_res_level=%s from registration_binning',
            scale_key)

    reg_sims = [sim1, sim2]

    # logging without use of %s
    logger.info("Registration resolution level: %s", scale_key)
    logger.info("Registration binning applied at loaded scale: %s", registration_binning)

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
            # xarray + dask fail if 'quality' is passed directly (?)
            "quality": xr.DataArray((da.ones(1) * quality)[0])
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


def _plot_registration_summaries(
    msims,
    transform_key,
    new_transform_key,
    g_reg_computed,
    groupwise_resolution_info_dict,
    show_plot,
):
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
        show_plot=show_plot,
    )

    fig_group_res, ax_group_res = None, None
    if groupwise_resolution_info_dict is not None:
        edge_residuals_dict = groupwise_resolution_info_dict.get(
            "edge_residuals"
        )
        if isinstance(edge_residuals_dict, dict):
            edge_residuals_dict = edge_residuals_dict.get(0, {})

        used_edges = groupwise_resolution_info_dict.get("used_edges")
        if isinstance(used_edges, dict):
            used_edges = used_edges.get(0, [])
        used_edge_set = (
            {tuple(sorted(e)) for e in used_edges}
            if used_edges
            else set()
        )

        if edge_residuals_dict is None or not hasattr(
            edge_residuals_dict, "get"
        ):
            edge_residuals_dict = {}

        edge_residuals = np.array(
            [
                edge_residuals_dict.get(tuple(sorted(e)), np.nan)
                for e in edges
            ]
        )
        edge_linestyles = [
            "-" if tuple(sorted(e)) in used_edge_set else ":"
            for e in edges
        ]
        finite_vals = edge_residuals[np.isfinite(edge_residuals)]
        if finite_vals.size == 0:
            edge_clims = [0, 1]
        else:
            vmin, vmax = np.nanmin(edge_residuals), np.nanmax(
                edge_residuals
            )
            edge_clims = [0, 1] if vmin == vmax else [vmin, vmax]
        fig_group_res, ax_group_res = vis_utils.plot_positions(
            msims,
            transform_key=new_transform_key,
            edges=edges,
            edge_color_vals=edge_residuals,
            edge_linestyles=edge_linestyles,
            edge_linestyle_labels={
                "-": "Used edges",
                ":": "Unused edges",
            },
            edge_cmap="Spectral_r",
            edge_clims=edge_clims,
            edge_label="Remaining edge residuals [distance units]",
            display_view_indices=True,
            use_positional_colors=False,
            plot_title="Global parameter resolution summary",
            show_plot=show_plot,
        )

    return {'fig_pair_reg': fig_pair_reg,
            'ax_pair_reg': ax_pair_reg,
            'fig_group_res': fig_group_res,
            'ax_group_res': ax_group_res}


def register(
    msims: list[MultiscaleSpatialImage],
    transform_key: str = None,
    reg_channel_index: int = None,
    reg_channel: str = None,
    new_transform_key: str = None,
    registration_binning: dict[str, int] = None,
    reg_res_level: int = None,
    overlap_tolerance: Union[float, dict[str, float]] = 0.0,
    pairwise_reg_func=phase_correlation_registration,
    pairwise_reg_func_kwargs: dict = None,
    groupwise_resolution_method="global_optimization",
    groupwise_resolution_kwargs: dict = None,
    pre_registration_pruning_method="alternating_pattern",
    pre_reg_pruning_method_kwargs: dict = None,
    post_registration_do_quality_filter: bool = False,
    post_registration_quality_threshold: float = 0.2,
    plot_summary: bool = False,
    pairs: list[tuple[int, int]] = None,
    scheduler=None,  # deprecated, see docstring
    n_parallel_pairwise_regs: int = None,
    return_dict: bool = False,
):
    """
    Register a list of views to a common extrinsic coordinate system.

    High-level flow:
    1) Build the overlap graph.
    2) Run pairwise registrations for selected edges.
    3) Resolve global transforms from the pairwise results.

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
        Binning applied to each dimension during registration, by default None.
        If reg_res_level is also provided, the binning factors must be compatible 
        with the resolution level.
    reg_res_level : int, optional
        Resolution level to use for registration (e.g., 0 for scale0, 1 for scale1).
        If None and registration_binning is provided, the optimal resolution level 
        is automatically determined. By default None.
    overlap_tolerance : float or dict, optional
        Extend overlap regions considered for pairwise registration.
        - if 0, the overlap region is the intersection of the bounding boxes.
        - if > 0, the overlap region is the intersection of the bounding boxes
            extended by this value in all spatial dimensions.
        - if None, the full images are used for registration
    pairwise_reg_func : Callable, optional
        Function used for registration. See the docs for the function API.
        By default, phase_correlation_registration is used. Another useful built-in
        registration function is `pairwise_reg_func=registration.registration_ANTsPy`
        for translation, rigid, similarity or affine registration using ANTsPy.
    pairwise_reg_func_kwargs : dict, optional
        Additional keyword arguments passed to the registration function.
        In the case of `pairwise_reg_func=registration_ANTsPy`, this can include e.g:
        - 'transform_type': ['Translation', 'Rigid' 'Affine'] or ['Similarity']
        For further parameters, see the docstring of the registration function.
    groupwise_resolution_method : str, optional
        Method used to resolve global transforms from pairwise registrations:
        - 'global_optimization' (transform: translation|rigid|similarity|affine)
        - 'shortest_paths' (uses the transform type defined by the pairwise registrations)
        - 'linear_two_pass' (transform: translation|rigid)
        Custom component-level methods can be registered via
        `param_resolution.register_groupwise_resolution_method(...)` and
        referenced by name.
    groupwise_resolution_kwargs : dict, optional
        Additional keyword arguments passed to the groupwise resolver.
        Parameters are method-specific. Common options include:
        - 'transform': final transform type (see method notes above)
        - 'reference_view': node index to keep fixed
        See the resolver docstrings for full details.
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
    pre_reg_pruning_method_kwargs : dict, optional
        Additional keyword arguments passed to the pre-registration pruning method, e.g.
        - 'keep_axis_aligned': 'max_angle' (larger angles between stack axis and pair edge are discarded, default 0.2)
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
           Solid edges were used by the resolver, dotted edges were unused.
        Stack boundary positions reflect the registration result.
        By default False
    pairs : list of tuples, optional
        If set, initialises the view adjacency graph using the indicates
        pairs of view/tile indices, by default None
    scheduler : str, optional
        (Deprecated since >0.1.28) Dask scheduler to use for parallel computation, by default None
        This parameter is deprecated and no longer used.
        Use a context manager instead to set the dask scheduler used within register(), e.g.
        `with dask.config.set(scheduler='threads'): register(...)`
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
            - 'metrics': Dictionary containing the following metrics:
                - 'edge_residuals': Dict[int, dict[tuple, float]] mapping timepoint index
                  to edge residuals
                - 'used_edges': Dict[int, list[tuple]] mapping timepoint index to edges
                  used by the resolution method
    """

    # warn about deprecated parameter
    if scheduler is not None:
        warnings.warn(
            "The register(..., scheduler=) parameter is deprecated, no longer used "
            "and will be removed in a future version. "
            "Use a context manager to set the dask scheduler used within register(), e.g. "
            "`with dask.config.set(scheduler='threads'): register(...)`",
            DeprecationWarning,
            stacklevel=2,
        )

    if pairwise_reg_func_kwargs is None:
        pairwise_reg_func_kwargs = {}

    if groupwise_resolution_kwargs is None:
        groupwise_resolution_kwargs = {}

    if pre_reg_pruning_method_kwargs is None:
        pre_reg_pruning_method_kwargs = {}

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
        g_reg = mv_graph.prune_view_adjacency_graph(
            g,
            method=pre_registration_pruning_method,
            pruning_method_kwargs=pre_reg_pruning_method_kwargs,
        )
    else:
        g_reg = g

    g_reg_computed = compute_pairwise_registrations(
        msims_reg,
        g_reg,
        transform_key=transform_key,
        registration_binning=registration_binning,
        reg_res_level=reg_res_level,
        overlap_tolerance=overlap_tolerance,
        pairwise_reg_func=pairwise_reg_func,
        pairwise_reg_func_kwargs=pairwise_reg_func_kwargs,
        n_parallel_pairwise_regs=n_parallel_pairwise_regs,
    )

    if post_registration_do_quality_filter:
        # filter edges by quality
        g_reg_computed = mv_graph.filter_edges(
            g_reg_computed,
            threshold=post_registration_quality_threshold,
            weight_key="quality",
        )

    params_dict, groupwise_resolution_info_dict = groupwise_resolution(
        g_reg_computed,
        method=groupwise_resolution_method,
        **groupwise_resolution_kwargs,
    )

    params = [
        params_dict[iview] for iview in sorted(g_reg_computed.nodes())
    ]

    if new_transform_key is not None:
        for imsim, msim in enumerate(msims):
            msi_utils.set_affine_transform(
                msim,
                params[imsim],
                transform_key=new_transform_key,
                base_transform_key=transform_key,
            )

    if plot_summary:
        plot_info = _plot_registration_summaries(
            msims,
            transform_key,
            new_transform_key,
            g_reg_computed,
            groupwise_resolution_info_dict,
            show_plot=plot_summary,
        )
    else:
        plot_info = {}

    if return_dict:
        return {
            "params": params,
            "pairwise_registration": {
                "graph": g_reg_computed,
                "metrics": {
                    "qualities": nx.get_edge_attributes(
                        g_reg_computed, "quality"
                    )
                },
                "summary_plot": None if plot_summary is False
                else (
                    plot_info['fig_pair_reg'],
                    plot_info['ax_pair_reg']
                    )
            },
            "groupwise_resolution": {
                "metrics": groupwise_resolution_info_dict,
                "summary_plot": None if plot_summary is False
                else (
                    plot_info['fig_group_res'],
                    plot_info['ax_group_res']
                )
            },
        }
    else:
        return params


def compute_pairwise_registrations(
    msims,
    g_reg,
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
        params = compute(params_xds)[0]
    elif n_parallel_pairwise_regs > 0:
        logger.info(
            "Pairwise registration(s) run in parallel: %s",
            n_parallel_pairwise_regs,
        )
        params = []
        for i in range(0, len(params_xds), n_parallel_pairwise_regs):
            params += compute(
                params_xds[i : i + n_parallel_pairwise_regs],
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
        fixed_data.data.astype(np.float32),
        origin=[fixed_origin[dim] for dim in spatial_dims][::-1],
        spacing=[fixed_spacing[dim] for dim in spatial_dims][::-1],
    )
    moving_ants = ants.from_numpy(
        moving_data.data.astype(np.float32),
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
