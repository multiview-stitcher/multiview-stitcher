import copy
import itertools
import logging
import math
import os
import tempfile
import warnings
from typing import Union

import dask.array as da
import networkx as nx
import numpy as np
import skimage.registration
import skimage.transform
import xarray as xr
from dask import compute, delayed
from dask.utils import has_keyword
from scipy import ndimage, stats
from scipy.spatial import cKDTree
from skimage.exposure import rescale_intensity
from skimage.metrics import structural_similarity

from multiview_stitcher.param_resolution import groupwise_resolution
from multiview_stitcher.transforms import AffineTransform, TranslationTransform

try:
    import ants
except ImportError:
    ants = None

itk = None  # lazy-imported inside registration_ITKElastix

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

MultiscaleSpatialImage = xr.DataTree


def _format_array_for_log(array, precision=4):
    return np.array2string(
        np.asarray(array),
        precision=precision,
        suppress_small=True,
    )


def _format_numeric_stats_for_log(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return "empty"

    return (
        f"min={np.min(values):.4g}, "
        f"median={np.median(values):.4g}, "
        f"p95={np.percentile(values, 95):.4g}, "
        f"max={np.max(values):.4g}"
    )


def _format_point_bounds_for_log(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or len(points) == 0:
        return "empty"

    return (
        "min="
        + _format_array_for_log(np.min(points, axis=0))
        + ", max="
        + _format_array_for_log(np.max(points, axis=0))
        + ", centroid="
        + _format_array_for_log(np.mean(points, axis=0))
    )


def _get_callable_name(func):
    return getattr(func, "__name__", func.__class__.__name__)


def _estimate_skimage_transform(transform_cls, fixed_points, moving_points, ndim):
    if hasattr(transform_cls, "from_estimate"):
        return transform_cls.from_estimate(fixed_points, moving_points)

    transform_model = transform_cls(dimensionality=ndim)
    if not transform_model.estimate(fixed_points, moving_points):
        return False

    return transform_model


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


def _get_overlap_bboxes(
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

    Return: dict with keys 'lowers' and 'uppers' containing the
    lower and upper bounds of overlap for both input images, as well
    as 'intersection' containing the intersection halfspace.
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

    vol, intersection = mv_graph.get_overlap_between_pair_of_stack_props(
        stack_props1=stack_propss[0],
        stack_props2=stack_propss[1],
    )

    corners = [
        mv_graph.get_vertices_from_stack_props(stack_props).reshape(-1, ndim)
        for stack_props in stack_propss
    ]

    corners = intersection.intersections

    if output_transform_key is None:
        # project corners into intrinsic coordinate system

        corners_intrinsic = [transformation.transform_pts(
                        corners,
                        np.linalg.inv(
                            spatial_image_utils.get_affine_from_sim(
                                sim, transform_key=input_transform_key
                            ).data
                        ),
                    )
            for sim in [sim1, sim2]]
        
        corners_target_space = corners_intrinsic

        # Transform the halfspace from world space (input_transform_key) into
        # the intrinsic coordinate space of sim1.  inv(T_sim1) maps world→intrinsic,
        # i.e. the same direction as transforming points into the new space.
        T_sim1 = spatial_image_utils.get_affine_from_sim(
            sim1, transform_key=input_transform_key
        ).data
        intersection = mv_graph.transform_halfspace(intersection, np.linalg.inv(T_sim1))

    elif output_transform_key == input_transform_key:
        corners_target_space = [corners, corners]
    else:
        raise NotImplementedError

    lowers = [np.min(cts, axis=0) for cts in corners_target_space]
    uppers = [np.max(cts, axis=0) for cts in corners_target_space]

    return {
        "lowers": lowers,
        "uppers": uppers,
        "intersection": intersection,
        "vol": vol,
    }


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


def _get_points_array_for_registration(point_set, sdims):
    position = point_set["position"]
    extra_dims = [
        dim for dim in position.dims if dim not in ["point_id", "dim"]
    ]
    for dim in extra_dims:
        if position.sizes[dim] != 1:
            raise ValueError(
                "Point sets passed to pairwise registration must be selected "
                f"to one value along dimension {dim!r}."
            )
        position = position.isel({dim: 0})

    position = position.sel(dim=sdims).transpose("point_id", "dim")

    points = np.asarray(position.data)
    valid = np.all(np.isfinite(points), axis=1)

    return points[valid]


def _transform_points_for_registration(points, affine):
    if points.size == 0:
        return points
    return transformation.transform_pts(points, affine)


def _get_marker_registration_min_matches(transform_type, ndim):
    transform_type = transform_type.lower()
    if transform_type == "translation":
        return 1
    if transform_type == "rigid":
        return ndim
    if transform_type == "affine":
        return ndim + 1
    raise ValueError(
        "Unsupported marker registration transform_type "
        f"{transform_type!r}. Expected 'translation', 'rigid', or 'affine'."
    )


def _get_marker_descriptor_vector_length(num_neighbors):
    return math.comb(num_neighbors + 1, 2)


def _get_marker_nearest_neighbor_scale(*point_sets):
    nearest_distances = []
    for points in point_sets:
        points = np.asarray(points, dtype=float)
        if len(points) < 2:
            continue
        distances, _ = cKDTree(points).query(points, k=2)
        nearest_distances.extend(distances[:, 1])

    nearest_distances = np.asarray(nearest_distances, dtype=float)
    nearest_distances = nearest_distances[np.isfinite(nearest_distances)]
    if nearest_distances.size == 0:
        return 0.0

    return float(np.median(nearest_distances))


def _get_marker_descriptor_distance_threshold(
    fixed_points,
    moving_points,
    num_neighbors,
    descriptor_threshold_scale,
):
    descriptor_vector_length = _get_marker_descriptor_vector_length(num_neighbors)
    nn_scale = _get_marker_nearest_neighbor_scale(fixed_points, moving_points)
    threshold = float(
        nn_scale * np.sqrt(descriptor_vector_length) * descriptor_threshold_scale
    )
    logger.debug(
        "Marker descriptor automatic threshold: nearest_neighbor_median=%.6g, "
        "descriptor_vector_length=%s, descriptor_threshold_scale=%.6g, "
        "threshold=%.6g",
        nn_scale,
        descriptor_vector_length,
        descriptor_threshold_scale,
        threshold,
    )
    return threshold


def _get_marker_descriptors(points, num_neighbors, redundancy, label=None):
    points = np.asarray(points, dtype=float)
    required_neighbors = num_neighbors + redundancy
    if len(points) < required_neighbors + 1:
        raise ValueError(
            "Not enough points to build marker descriptors. "
            f"Need at least {required_neighbors + 1}, got {len(points)}."
        )

    tree = cKDTree(points)
    query_k = min(len(points), required_neighbors + 2)
    _, neighbor_indices = tree.query(points, k=query_k)

    descriptors = []
    for point_index, point_neighbor_indices in enumerate(neighbor_indices):
        point_neighbor_indices = np.atleast_1d(point_neighbor_indices)
        point_neighbor_indices = [
            int(ind) for ind in point_neighbor_indices if int(ind) != point_index
        ][:required_neighbors]

        if len(point_neighbor_indices) < required_neighbors:
            continue

        for subset in itertools.combinations(point_neighbor_indices, num_neighbors):
            descriptor_points = points[[point_index] + list(subset)]
            distances = []
            for i, j in itertools.combinations(
                range(len(descriptor_points)), 2
            ):
                distances.append(
                    np.linalg.norm(descriptor_points[i] - descriptor_points[j])
                )
            descriptors.append(
                {
                    "point_index": point_index,
                    "vector": np.sort(np.asarray(distances, dtype=float)),
                }
            )

    if len(descriptors) == 0:
        raise ValueError("No marker descriptors could be built.")

    logger.debug(
        "%s marker descriptors: points=%s, num_neighbors=%s, redundancy=%s, "
        "required_neighbors=%s, descriptors=%s, descriptors_per_point=%s, "
        "descriptor_vector_length=%s",
        label.capitalize() if label else "Built",
        len(points),
        num_neighbors,
        redundancy,
        required_neighbors,
        len(descriptors),
        math.comb(required_neighbors, num_neighbors),
        len(descriptors[0]["vector"]),
    )
    return descriptors


def _match_marker_descriptors(
    fixed_descriptors,
    moving_descriptors,
    descriptor_ratio,
    descriptor_distance_threshold,
):
    fixed_vectors = np.asarray(
        [descriptor["vector"] for descriptor in fixed_descriptors], dtype=float
    )
    fixed_point_indices = np.asarray(
        [descriptor["point_index"] for descriptor in fixed_descriptors],
        dtype=int,
    )
    moving_vectors = np.asarray(
        [descriptor["vector"] for descriptor in moving_descriptors],
        dtype=float,
    )
    moving_point_indices = np.asarray(
        [descriptor["point_index"] for descriptor in moving_descriptors],
        dtype=int,
    )

    if len(fixed_vectors) == 0 or len(moving_vectors) == 0:
        return np.empty((0, 2), dtype=int)

    _, moving_descriptor_counts = np.unique(
        moving_point_indices, return_counts=True
    )
    query_k = min(
        len(moving_vectors),
        int(np.max(moving_descriptor_counts)) + 1,
    )
    descriptor_tree = cKDTree(moving_vectors)
    nearest_distances, nearest_indices = descriptor_tree.query(
        fixed_vectors, k=query_k
    )
    nearest_distances = np.asarray(nearest_distances, dtype=float)
    nearest_indices = np.asarray(nearest_indices, dtype=int)
    if query_k == 1:
        nearest_distances = nearest_distances[:, np.newaxis]
        nearest_indices = nearest_indices[:, np.newaxis]

    candidates_by_pair = {}
    accepted_descriptor_matches = 0
    rejected_by_distance = 0
    rejected_by_ratio = 0

    for fixed_point_index, row_distances, row_indices in zip(
        fixed_point_indices,
        nearest_distances,
        nearest_indices,
    ):
        best_descriptor_index = row_indices[0]
        best_moving_point_index = moving_point_indices[best_descriptor_index]
        best_distance = float(row_distances[0])
        passes_distance = best_distance < descriptor_distance_threshold
        if not passes_distance:
            rejected_by_distance += 1
            continue

        row_moving_point_indices = moving_point_indices[row_indices]
        second_best_mask = row_moving_point_indices != best_moving_point_index
        if np.any(second_best_mask):
            second_best_distance = float(
                row_distances[np.flatnonzero(second_best_mask)[0]]
            )
        else:
            second_best_distance = np.inf

        passes_ratio = best_distance * descriptor_ratio < second_best_distance
        if passes_ratio:
            accepted_descriptor_matches += 1
            pair = (fixed_point_index, best_moving_point_index)
            if (
                pair not in candidates_by_pair
                or best_distance < candidates_by_pair[pair]
            ):
                candidates_by_pair[pair] = best_distance
        else:
            rejected_by_ratio += 1

    candidate_pairs = np.asarray(list(candidates_by_pair.keys()), dtype=int)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Marker descriptor matching: fixed_descriptors=%s, moving_descriptors=%s, "
            "descriptor_ratio=%.6g, descriptor_distance_threshold=%.6g, "
            "accepted_descriptor_matches=%s, unique_candidate_pairs=%s, "
            "rejected_by_distance=%s, rejected_by_ratio=%s, "
            "candidate_distance_stats=%s",
            len(fixed_descriptors),
            len(moving_descriptors),
            descriptor_ratio,
            descriptor_distance_threshold,
            accepted_descriptor_matches,
            len(candidate_pairs),
            rejected_by_distance,
            rejected_by_ratio,
            _format_numeric_stats_for_log(list(candidates_by_pair.values())),
        )
        if len(candidate_pairs):
            logger.debug(
                "Marker descriptor candidate pair sample: %s",
                _format_array_for_log(candidate_pairs[:10]),
            )

    return candidate_pairs


def _fit_marker_transform(fixed_points, moving_points, transform_type):
    fixed_points = np.asarray(fixed_points, dtype=float)
    moving_points = np.asarray(moving_points, dtype=float)
    ndim = fixed_points.shape[1]
    transform_type = transform_type.lower()

    if transform_type == "translation":
        translation = np.mean(moving_points - fixed_points, axis=0)
        transform_model = skimage.transform.EuclideanTransform(
            translation=translation,
            dimensionality=ndim,
        )
        return np.asarray(transform_model.params, dtype=float)

    if transform_type == "rigid":
        transform_model = _estimate_skimage_transform(
            skimage.transform.EuclideanTransform,
            fixed_points,
            moving_points,
            ndim,
        )
        if not transform_model:
            raise ValueError(
                "Rigid marker registration points are degenerate. "
                f"skimage.transform returned: {transform_model}"
            )
        return np.asarray(transform_model.params, dtype=float)

    if transform_type == "affine":
        transform_model = _estimate_skimage_transform(
            skimage.transform.AffineTransform,
            fixed_points,
            moving_points,
            ndim,
        )
        if not transform_model:
            raise ValueError(
                "Affine marker registration points are degenerate. "
                f"skimage.transform returned: {transform_model}"
            )
        return np.asarray(transform_model.params, dtype=float)

    raise ValueError(
        "Unsupported marker registration transform_type "
        f"{transform_type!r}. Expected 'translation', 'rigid', or 'affine'."
    )


def _score_marker_transform(affine, fixed_points, moving_points, ransac_max_error):
    transformed_fixed_points = transformation.transform_pts(fixed_points, affine)
    residuals = np.linalg.norm(transformed_fixed_points - moving_points, axis=1)
    inlier_mask = residuals <= ransac_max_error
    return residuals, inlier_mask


def _run_marker_ransac(
    fixed_points,
    moving_points,
    candidate_pairs,
    transform_type,
    ransac_max_error,
    ransac_min_inlier_ratio,
    ransac_min_inlier_factor,
    ransac_num_iterations,
    random_state,
):
    ndim = fixed_points.shape[1]
    min_model_matches = _get_marker_registration_min_matches(
        transform_type, ndim
    )
    min_inliers = max(
        min_model_matches,
        int(np.round(min_model_matches * ransac_min_inlier_factor)),
    )

    if len(candidate_pairs) < min_inliers:
        logger.debug(
            "Marker RANSAC skipped: candidate_pairs=%s, min_inliers=%s, "
            "transform_type=%s, ndim=%s",
            len(candidate_pairs),
            min_inliers,
            transform_type,
            ndim,
        )
        raise ValueError(
            "Not enough marker correspondences for RANSAC. "
            f"Need at least {min_inliers}, got {len(candidate_pairs)}."
        )

    fixed_candidates = fixed_points[candidate_pairs[:, 0]]
    moving_candidates = moving_points[candidate_pairs[:, 1]]
    rng = np.random.default_rng(random_state)
    best_result = None
    num_candidates = len(candidate_pairs)
    num_combinations = math.comb(num_candidates, min_model_matches)

    if num_combinations <= ransac_num_iterations:
        sample_iter = itertools.combinations(
            range(num_candidates), min_model_matches
        )
        sampling_mode = "exhaustive"
        num_samples = num_combinations
    else:
        sample_iter = (
            rng.choice(
                num_candidates, size=min_model_matches, replace=False
            )
            for _ in range(ransac_num_iterations)
        )
        sampling_mode = "random"
        num_samples = ransac_num_iterations

    logger.debug(
        "Marker RANSAC start: transform_type=%s, ndim=%s, candidate_pairs=%s, "
        "min_model_matches=%s, min_inliers=%s, min_inlier_ratio=%.6g, "
        "max_error=%.6g, sampling_mode=%s, samples=%s, possible_combinations=%s",
        transform_type,
        ndim,
        num_candidates,
        min_model_matches,
        min_inliers,
        ransac_min_inlier_ratio,
        ransac_max_error,
        sampling_mode,
        num_samples,
        num_combinations,
    )

    for sample_indices in sample_iter:
        sample_indices = np.asarray(sample_indices, dtype=int)
        try:
            affine = _fit_marker_transform(
                fixed_candidates[sample_indices],
                moving_candidates[sample_indices],
                transform_type,
            )
        except ValueError:
            continue

        residuals, inlier_mask = _score_marker_transform(
            affine,
            fixed_candidates,
            moving_candidates,
            ransac_max_error,
        )
        num_inliers = int(np.sum(inlier_mask))
        if num_inliers == 0:
            mean_residual = np.inf
            model_quality = 0.0
        else:
            mean_residual = float(np.mean(residuals[inlier_mask]))
            model_quality = (num_inliers / num_candidates) * max(
                0.0, 1.0 - mean_residual / ransac_max_error
            )

        result_key = (model_quality, num_inliers, -mean_residual)
        if best_result is None or result_key > best_result["key"]:
            best_result = {
                "key": result_key,
                "inlier_mask": inlier_mask,
                "mean_residual": mean_residual,
                "quality": model_quality,
            }

    if best_result is None:
        logger.debug("Marker RANSAC failed: no non-degenerate sample produced a model.")
        raise ValueError("No marker transform model could be estimated.")

    inlier_mask = best_result["inlier_mask"]
    num_inliers = int(np.sum(inlier_mask))
    inlier_ratio = num_inliers / num_candidates
    logger.debug(
        "Marker RANSAC best model before refit: inliers=%s/%s, "
        "inlier_ratio=%.4f, mean_inlier_residual=%.4g, quality=%.4f",
        num_inliers,
        num_candidates,
        inlier_ratio,
        best_result["mean_residual"],
        best_result["quality"],
    )

    if num_inliers < min_inliers or inlier_ratio < ransac_min_inlier_ratio:
        logger.debug(
            "Marker RANSAC rejected best model: inliers=%s/%s, "
            "inlier_ratio=%.4f, required_min_inliers=%s, "
            "required_min_inlier_ratio=%.4f",
            num_inliers,
            num_candidates,
            inlier_ratio,
            min_inliers,
            ransac_min_inlier_ratio,
        )
        raise ValueError(
            "Marker RANSAC did not find enough inliers. "
            f"Found {num_inliers}/{num_candidates} inliers."
        )

    affine = _fit_marker_transform(
        fixed_candidates[inlier_mask],
        moving_candidates[inlier_mask],
        transform_type,
    )
    residuals, inlier_mask = _score_marker_transform(
        affine,
        fixed_candidates,
        moving_candidates,
        ransac_max_error,
    )
    num_inliers = int(np.sum(inlier_mask))
    if num_inliers < min_inliers:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Marker RANSAC refit rejected: inliers=%s/%s, "
                "required_min_inliers=%s, residual_stats=%s",
                num_inliers,
                num_candidates,
                min_inliers,
                _format_numeric_stats_for_log(residuals),
            )
        raise ValueError(
            "Refit marker transform did not preserve enough inliers. "
            f"Found {num_inliers}/{num_candidates} inliers."
        )
    mean_residual = float(np.mean(residuals[inlier_mask]))
    inlier_ratio = float(num_inliers / num_candidates)
    quality = inlier_ratio * max(0.0, 1.0 - mean_residual / ransac_max_error)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Marker RANSAC refit accepted: inliers=%s/%s, inlier_ratio=%.4f, "
            "mean_inlier_residual=%.4g, quality=%.4f, residual_stats=%s, "
            "inlier_residual_stats=%s, affine=\n%s",
            num_inliers,
            num_candidates,
            inlier_ratio,
            mean_residual,
            quality,
            _format_numeric_stats_for_log(residuals),
            _format_numeric_stats_for_log(residuals[inlier_mask]),
            _format_array_for_log(affine),
        )

    return affine, quality


def _run_marker_icp(
    fixed_points,
    moving_points,
    initial_affine,
    initial_quality,
    transform_type,
    icp_max_error,
    icp_num_iterations,
    icp_tolerance,
):
    fixed_points = np.asarray(fixed_points, dtype=float)
    moving_points = np.asarray(moving_points, dtype=float)
    affine = np.asarray(initial_affine, dtype=float)
    ndim = fixed_points.shape[1]
    min_matches = _get_marker_registration_min_matches(transform_type, ndim)
    moving_tree = cKDTree(moving_points)
    quality = float(initial_quality)

    logger.debug(
        "Marker ICP start: transform_type=%s, fixed_points=%s, moving_points=%s, "
        "max_error=%.6g, iterations=%s, tolerance=%.6g",
        transform_type,
        len(fixed_points),
        len(moving_points),
        icp_max_error,
        icp_num_iterations,
        icp_tolerance,
    )

    for iteration in range(icp_num_iterations):
        transformed_fixed_points = transformation.transform_pts(fixed_points, affine)
        nearest_distances, nearest_indices = moving_tree.query(
            transformed_fixed_points, k=1
        )
        inlier_mask = nearest_distances <= icp_max_error
        num_inliers = int(np.sum(inlier_mask))
        if num_inliers < min_matches:
            logger.debug(
                "Marker ICP stopped: iteration=%s, inliers=%s, min_matches=%s",
                iteration,
                num_inliers,
                min_matches,
            )
            break

        # Fit the requested model to nearest-neighbor marker correspondences,
        # always from the original fixed points to avoid accumulating drift.
        try:
            next_affine = _fit_marker_transform(
                fixed_points[inlier_mask],
                moving_points[nearest_indices[inlier_mask]],
                transform_type,
            )
        except ValueError as exc:
            logger.debug(
                "Marker ICP stopped: iteration=%s, fit failed: %s",
                iteration,
                exc,
            )
            break

        mean_residual = float(np.mean(nearest_distances[inlier_mask]))
        quality = (num_inliers / len(fixed_points)) * max(
            0.0, 1.0 - mean_residual / icp_max_error
        )
        affine_delta = float(np.linalg.norm(next_affine - affine))
        affine = next_affine

        logger.debug(
            "Marker ICP iteration %s: inliers=%s/%s, mean_residual=%.4g, "
            "quality=%.4f, affine_delta=%.4g",
            iteration + 1,
            num_inliers,
            len(fixed_points),
            mean_residual,
            quality,
            affine_delta,
        )
        if affine_delta <= icp_tolerance:
            break

    return affine, quality


def _fail_marker_registration(ndim, message, fail_on_error):
    logger.debug(
        "Marker registration failed: ndim=%s, fail_on_error=%s, message=%s",
        ndim,
        fail_on_error,
        message,
    )
    if fail_on_error:
        raise ValueError(message)

    warnings.warn(message, UserWarning, stacklevel=2)
    return {
        "affine_matrix": np.eye(ndim + 1),
        "quality": np.nan,
    }


def registration_marker_based(
    fixed_points,
    moving_points,
    transform_type="rigid",
    num_neighbors=3,
    redundancy=1,
    descriptor_ratio=3.0,
    descriptor_distance_threshold=None,
    descriptor_threshold_scale=1.0,
    ransac_max_error=5.0,
    ransac_min_inlier_ratio=0.1,
    ransac_min_inlier_factor=3.0,
    ransac_num_iterations=1000,
    icp=False,
    icp_max_error=None,
    icp_num_iterations=50,
    icp_tolerance=1e-6,
    random_state=0,
    fail_on_error=True,
):
    """
    Marker-based registration inspired by BigStitcher RGLDM bead matching.

    The function matches local geometric descriptors computed from
    ``fixed_points`` and ``moving_points``, removes inconsistent matches with
    RANSAC, and returns a homogeneous transform from fixed to moving points.

    Parameters
    ----------
    fixed_points, moving_points : array-like
        Point coordinates with shape ``(n_points, n_spatial_dims)``.
    transform_type : {"translation", "rigid", "affine"}, optional
        Transformation model to estimate. By default "rigid".
    num_neighbors : int, optional
        Number of nearest neighbors used for each local descriptor.
    redundancy : int, optional
        Additional neighbors considered when forming descriptor subsets.
    descriptor_ratio : float, optional
        Best match must be this many times better than the second-best match.
    descriptor_distance_threshold : float, optional
        Maximum descriptor distance. If None, it is estimated from nearest
        neighbor distances in the point sets.
    descriptor_threshold_scale : float, optional
        Scale factor for the automatic descriptor distance threshold.
    ransac_max_error : float, optional
        Maximum point residual for a correspondence to count as an inlier.
    ransac_min_inlier_ratio : float, optional
        Minimum fraction of candidate correspondences that must be inliers.
    ransac_min_inlier_factor : float, optional
        Multiplier for the model's minimum number of required matches.
    ransac_num_iterations : int, optional
        Number of random RANSAC samples.
    icp : bool, optional
        If True, refine the RANSAC result with nearest-neighbor ICP.
    icp_max_error : float, optional
        Maximum nearest-neighbor distance for ICP correspondences. If None,
        ``ransac_max_error`` is used.
    icp_num_iterations : int, optional
        Maximum number of ICP refinement iterations.
    icp_tolerance : float, optional
        Stop ICP when the affine matrix update has this Frobenius norm or less.
    random_state : int or numpy.random.Generator, optional
        Seed or generator used for RANSAC sampling.
    fail_on_error : bool, optional
        If True, raise on failure. Otherwise warn and return identity with
        ``quality=np.nan``.

    References
    ----------
    BigStitcher/Fiji RGLDM bead registration:
    https://github.com/fiji/SPIM_Registration/tree/master/src/main/java/spim/process/interestpointregistration/geometricdescriptor
    """

    fixed_points = np.asarray(fixed_points, dtype=float)
    moving_points = np.asarray(moving_points, dtype=float)
    if fixed_points.ndim == 2:
        ndim = fixed_points.shape[1]
    elif moving_points.ndim == 2:
        ndim = moving_points.shape[1]
    else:
        ndim = 2

    try:
        logger.debug(
            "Marker registration input: fixed_shape=%s, moving_shape=%s, "
            "transform_type=%s, num_neighbors=%s, redundancy=%s, "
            "descriptor_ratio=%.6g, descriptor_distance_threshold=%s, "
            "descriptor_threshold_scale=%.6g, ransac_max_error=%.6g, "
            "ransac_min_inlier_ratio=%.6g, ransac_min_inlier_factor=%.6g, "
            "ransac_num_iterations=%s, icp=%s, icp_max_error=%s, "
            "icp_num_iterations=%s, icp_tolerance=%.6g, random_state=%s",
            fixed_points.shape,
            moving_points.shape,
            transform_type,
            num_neighbors,
            redundancy,
            descriptor_ratio,
            descriptor_distance_threshold,
            descriptor_threshold_scale,
            ransac_max_error,
            ransac_min_inlier_ratio,
            ransac_min_inlier_factor,
            ransac_num_iterations,
            icp,
            icp_max_error,
            icp_num_iterations,
            icp_tolerance,
            random_state,
        )
        if fixed_points.ndim != 2 or moving_points.ndim != 2:
            raise ValueError("Marker point arrays must be two-dimensional.")
        if fixed_points.shape[1] != moving_points.shape[1]:
            raise ValueError(
                "Fixed and moving marker points must have the same dimensionality."
            )
        if not len(fixed_points) or not len(moving_points):
            raise ValueError("Marker point arrays must not be empty.")
        if num_neighbors < 1:
            raise ValueError("num_neighbors must be at least 1.")
        if redundancy < 0:
            raise ValueError("redundancy must be non-negative.")
        if descriptor_ratio <= 0:
            raise ValueError("descriptor_ratio must be positive.")
        if descriptor_threshold_scale < 0:
            raise ValueError("descriptor_threshold_scale must be non-negative.")
        if ransac_max_error <= 0:
            raise ValueError("ransac_max_error must be positive.")
        if ransac_num_iterations < 1:
            raise ValueError("ransac_num_iterations must be at least 1.")
        if icp_max_error is None:
            icp_max_error = ransac_max_error
        elif icp_max_error <= 0:
            raise ValueError("icp_max_error must be positive.")
        if icp_num_iterations < 1:
            raise ValueError("icp_num_iterations must be at least 1.")
        if icp_tolerance < 0:
            raise ValueError("icp_tolerance must be non-negative.")

        transform_type = str(transform_type).lower()
        _get_marker_registration_min_matches(transform_type, ndim)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Marker registration point extents: fixed=(%s), moving=(%s)",
                _format_point_bounds_for_log(fixed_points),
                _format_point_bounds_for_log(moving_points),
            )

        if descriptor_distance_threshold is None:
            descriptor_distance_threshold = (
                _get_marker_descriptor_distance_threshold(
                    fixed_points,
                    moving_points,
                    num_neighbors,
                    descriptor_threshold_scale,
                )
            )
        elif descriptor_distance_threshold < 0:
            raise ValueError("descriptor_distance_threshold must be non-negative.")
        else:
            logger.debug(
                "Marker descriptor manual threshold: threshold=%.6g",
                descriptor_distance_threshold,
            )

        fixed_descriptors = _get_marker_descriptors(
            fixed_points, num_neighbors, redundancy, label="fixed"
        )
        moving_descriptors = _get_marker_descriptors(
            moving_points, num_neighbors, redundancy, label="moving"
        )
        candidate_pairs = _match_marker_descriptors(
            fixed_descriptors,
            moving_descriptors,
            descriptor_ratio,
            descriptor_distance_threshold,
        )
        if len(candidate_pairs) == 0:
            raise ValueError("No marker correspondence candidates found.")

        affine, quality = _run_marker_ransac(
            fixed_points,
            moving_points,
            candidate_pairs,
            transform_type,
            ransac_max_error,
            ransac_min_inlier_ratio,
            ransac_min_inlier_factor,
            ransac_num_iterations,
            random_state,
        )
        if icp:
            affine, quality = _run_marker_icp(
                fixed_points,
                moving_points,
                affine,
                quality,
                transform_type,
                icp_max_error,
                icp_num_iterations,
                icp_tolerance,
            )

    except ValueError as exc:
        return _fail_marker_registration(ndim, str(exc), fail_on_error)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Marker registration result: quality=%.4f, fixed_to_moving_affine=\n%s",
            quality,
            _format_array_for_log(affine),
        )
    return {
        "affine_matrix": affine,
        "quality": quality,
    }


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
    pairwise_reg_func,
    fixed_data=None,
    moving_data=None,
    skip_constant_check=False,
    **pairwise_reg_func_kwargs,
):
    """
    Check that images are not constant and apply the registration function.
    """
    has_image_data = fixed_data is not None and moving_data is not None
    func_name = _get_callable_name(pairwise_reg_func)
    fixed_points = pairwise_reg_func_kwargs.get("fixed_points")
    moving_points = pairwise_reg_func_kwargs.get("moving_points")
    logger.debug(
        "Dispatching pairwise registration: func=%s, has_image_data=%s, "
        "fixed_data_shape=%s, moving_data_shape=%s, skip_constant_check=%s, "
        "fixed_points_shape=%s, moving_points_shape=%s, kwargs=%s",
        func_name,
        has_image_data,
        np.shape(fixed_data) if fixed_data is not None else None,
        np.shape(moving_data) if moving_data is not None else None,
        skip_constant_check,
        np.shape(fixed_points) if fixed_points is not None else None,
        np.shape(moving_points) if moving_points is not None else None,
        sorted(pairwise_reg_func_kwargs.keys()),
    )
    if has_image_data and not skip_constant_check:
        int_extrema = [
            [func(im) for im in [fixed_data, moving_data]]
            for func in [np.nanmin, np.nanmax]
        ]

        # return if no translation if images are constant
        for i in range(2):
            if int_extrema[0][i] == int_extrema[1][i]:
                logger.debug(
                    "Pairwise registration skipped because %s image is constant: "
                    "func=%s, min=max=%s",
                    ["fixed", "moving"][i],
                    func_name,
                    int_extrema[0][i],
                )
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

    if has_image_data:
        pairwise_reg_func_kwargs["fixed_data"] = fixed_data
        pairwise_reg_func_kwargs["moving_data"] = moving_data

    reg_result = pairwise_reg_func(**pairwise_reg_func_kwargs)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Pairwise registration returned: func=%s, quality=%s, affine=\n%s",
            func_name,
            reg_result.get("quality"),
            _format_array_for_log(reg_result.get("affine_matrix")),
        )
    return reg_result


def register_pair_of_msims(
    msim1,
    msim2,
    transform_key,
    points_key="beads",
    prefilter_markers: bool = False,
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
    points_key : str, optional
        Named point set to pass to marker-aware pairwise registration functions.
    prefilter_markers : bool, optional
        If True, restrict markers to the pairwise overlap before passing them
        to marker-aware pairwise registration functions. By default False.
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
    pairwise_reg_func_name = _get_callable_name(pairwise_reg_func)
    logger.debug(
        "Registering pair of msims: func=%s, transform_key=%s, points_key=%s, "
        "prefilter_markers=%s, spatial_dims=%s, registration_binning=%s, "
        "reg_res_level=%s, overlap_tolerance=%s",
        pairwise_reg_func_name,
        transform_key,
        points_key,
        prefilter_markers,
        spatial_dims,
        registration_binning,
        reg_res_level,
        overlap_tolerance,
    )

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
                str(registration_binning))

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
    logger.info("Registration binning applied at loaded scale: %s", str(registration_binning))

    # Ensure dask-backed up front so that all subsequent operations
    # (coarsen, sel, transform) stay lazy for zarr-backed inputs.
    reg_sims = [
        spatial_image_utils.ensure_dask_backed_dataarray(sim)
        for sim in reg_sims
    ]

    if max(registration_binning.values()) > 1:
        reg_sims_b = []
        for sim in reg_sims:
            sim_binned = (
                sim.coarsen(registration_binning, boundary="trim")
                .mean()
                .astype(sim.dtype)
            )
            sim_binned.attrs.update(copy.deepcopy(sim.attrs))
            reg_sims_b.append(sim_binned)
    else:
        reg_sims_b = reg_sims

    reg_sims_b_marker_source = reg_sims_b

    overlap_dict = _get_overlap_bboxes(
        reg_sims_b[0],
        reg_sims_b[1],
        input_transform_key=transform_key,
        output_transform_key=None,
        overlap_tolerance=overlap_tolerance,
    )
    lowers, uppers = overlap_dict["lowers"], overlap_dict["uppers"]
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Pairwise registration overlap in loaded coordinate space: "
            "lowers=%s, uppers=%s",
            _format_array_for_log(lowers),
            _format_array_for_log(uppers),
        )

    reg_sims_spacing = [
        spatial_image_utils.get_spacing_from_sim(sim) for sim in reg_sims_b
    ]

    tol = 1e-6
    reg_sims_b = [
        spatial_image_utils.sim_sel_coords(
            sim,
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
    pairwise_reg_func_has_data_keywords = {
        keyword: has_keyword(pairwise_reg_func, keyword)
        for keyword in ["fixed_data", "moving_data"]
    }
    pairwise_reg_func_has_point_keywords = {
        keyword: has_keyword(pairwise_reg_func, keyword)
        for keyword in ["fixed_points", "moving_points"]
    }

    if np.any(list(pairwise_reg_func_has_data_keywords.values())) and not np.all(
        list(pairwise_reg_func_has_data_keywords.values())
    ):
        raise ValueError(
            "Image-aware pairwise registration functions must accept both "
            "'fixed_data' and 'moving_data'."
        )

    pairwise_reg_func_has_image_data = np.all(
        list(pairwise_reg_func_has_data_keywords.values())
    )

    if np.any(list(pairwise_reg_func_has_point_keywords.values())) and not np.all(
        list(pairwise_reg_func_has_point_keywords.values())
    ):
        raise ValueError(
            "Point-aware pairwise registration functions must accept both "
            "'fixed_points' and 'moving_points'."
        )

    if np.all(list(pairwise_reg_func_has_point_keywords.values())):
        registration_func_space = "transform_key_space"

        affines = [
            spatial_image_utils.get_affine_from_sim(
                sim, transform_key=transform_key
            )
            .squeeze()
            .data
            for sim in reg_sims_b
        ]
        initial_affine = np.matmul(np.linalg.inv(affines[1]), affines[0])
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Point-aware pairwise registration setup: func=%s, "
                "registration_func_space=%s, accepts_image_data=%s, "
                "initial_affine=\n%s",
                pairwise_reg_func_name,
                registration_func_space,
                pairwise_reg_func_has_image_data,
                _format_array_for_log(initial_affine),
            )

        marker_sims = reg_sims_b if prefilter_markers else reg_sims_b_marker_source
        fixed_point_set = spatial_image_utils.get_point_set(
            marker_sims[0],
            points_key=points_key,
        )
        moving_point_set = spatial_image_utils.get_point_set(
            marker_sims[1],
            points_key=points_key,
        )
        fixed_points = _get_points_array_for_registration(
            fixed_point_set,
            spatial_dims,
        )
        moving_points = _get_points_array_for_registration(
            moving_point_set,
            spatial_dims,
        )

        fixed_points_for_registration = _transform_points_for_registration(
            fixed_points,
            affines[0],
        )
        moving_points_for_registration = _transform_points_for_registration(
            moving_points,
            affines[1],
        )
        pairwise_reg_func_kwargs["fixed_points"] = fixed_points_for_registration
        pairwise_reg_func_kwargs["moving_points"] = moving_points_for_registration

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Marker points passed to pairwise registration: points_key=%r, "
                "prefilter_markers=%s, fixed_count=%s, moving_count=%s, "
                "fixed_bounds=(%s), moving_bounds=(%s)",
                points_key,
                prefilter_markers,
                len(fixed_points_for_registration),
                len(moving_points_for_registration),
                _format_point_bounds_for_log(fixed_points_for_registration),
                _format_point_bounds_for_log(moving_points_for_registration),
            )

        for isim, sim in enumerate(reg_sims_b):
            prefix = ["fixed", "moving"][isim]
            if pairwise_reg_func_has_keywords[f"{prefix}_origin"]:
                pairwise_reg_func_kwargs[
                    f"{prefix}_origin"
                ] = spatial_image_utils.get_origin_from_sim(sim)
            if pairwise_reg_func_has_keywords[f"{prefix}_spacing"]:
                pairwise_reg_func_kwargs[
                    f"{prefix}_spacing"
                ] = spatial_image_utils.get_spacing_from_sim(sim)

        if pairwise_reg_func_has_keywords["initial_affine"]:
            pairwise_reg_func_kwargs[
                "initial_affine"
            ] = param_utils.affine_to_xaffine(initial_affine)

        fixed_data = None
        moving_data = None
        if pairwise_reg_func_has_image_data:
            fixed_data = reg_sims_b[0].data
            moving_data = reg_sims_b[1].data

    elif not np.any(list(pairwise_reg_func_has_keywords.values())):
        fixed_data = None
        moving_data = None

        if pairwise_reg_func_has_image_data:
            registration_func_space = "pixel_space"

            sims_pixel_space = sims_to_intrinsic_coord_system(
                reg_sims_b[0],
                reg_sims_b[1],
                transform_key=transform_key,
                overlap_bboxes=(lowers, uppers),
            )

            fixed_data = sims_pixel_space[0].data
            moving_data = sims_pixel_space[1].data
        else:
            registration_func_space = "transform_key_space"

    elif np.all(list(pairwise_reg_func_has_keywords.values())):
        registration_func_space = "physical_space"

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

        fixed_data = None
        moving_data = None
        if pairwise_reg_func_has_image_data:
            fixed_data = reg_sims_b[0].data
            moving_data = reg_sims_b[1].data

    else:
        raise ValueError("Unknown registration function signature")

    logger.debug(
        "Pairwise registration call prepared: func=%s, registration_func_space=%s, "
        "accepts_image_data=%s, accepts_points=%s, kwargs=%s",
        pairwise_reg_func_name,
        registration_func_space,
        pairwise_reg_func_has_image_data,
        np.all(list(pairwise_reg_func_has_point_keywords.values())),
        sorted(pairwise_reg_func_kwargs.keys()),
    )

    if pairwise_reg_func_has_image_data:
        fixed_data = xr.DataArray(fixed_data, dims=spatial_dims)
        moving_data = xr.DataArray(moving_data, dims=spatial_dims)

    param_dict_d = delayed(dispatch_pairwise_reg_func, nout=1)(
        pairwise_reg_func,
        fixed_data=fixed_data,
        moving_data=moving_data,
        skip_constant_check=(
            not pairwise_reg_func_has_image_data
            or registration_func_space == "transform_key_space"
        ),
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
    elif registration_func_space == "transform_key_space":
        affine_phys = affine

    param_ds = xr.Dataset(
        data_vars={
            "transform": param_utils.affine_to_xaffine(affine_phys),
            # xarray + dask fail if 'quality' is passed directly (?)
            "quality": xr.DataArray((da.ones(1) * quality)[0])
        }
    )

    # attach bbox in physical coordinates
    overlap_dict = _get_overlap_bboxes(
        sim1,
        sim2,
        input_transform_key=transform_key,
        output_transform_key=transform_key,
        overlap_tolerance=overlap_tolerance,
    )
    lowers_phys, uppers_phys = overlap_dict["lowers"], overlap_dict["uppers"]

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
    points_key: str = "beads",
    prefilter_markers: bool = False,
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
    points_key : str, optional
        Named point set to use for marker-aware pairwise registration functions.
    prefilter_markers : bool, optional
        If True, restrict markers to each pairwise overlap before marker-based
        pairwise registration. By default False.
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

    # determine registration pairs from input images
    g = mv_graph.build_view_adjacency_graph_from_msims(
        msims_reg,
        transform_key=transform_key,
        pairs=pairs,
        overlap_tolerance=overlap_tolerance,
    )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Registration graph built: nodes=%s, edges=%s, edge_list=%s, "
            "transform_key=%s, overlap_tolerance=%s",
            g.number_of_nodes(),
            g.number_of_edges(),
            sorted(g.edges()),
            transform_key,
            overlap_tolerance,
        )

    # prune registration pair graph
    if pre_registration_pruning_method is not None:
        g_reg = mv_graph.prune_view_adjacency_graph(
            g,
            method=pre_registration_pruning_method,
            pruning_method_kwargs=pre_reg_pruning_method_kwargs,
        )
    else:
        g_reg = g
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Registration graph after pruning: method=%s, nodes=%s, edges=%s, "
            "edge_list=%s",
            pre_registration_pruning_method,
            g_reg.number_of_nodes(),
            g_reg.number_of_edges(),
            sorted(g_reg.edges()),
        )

    # if required, import itk already here
    # to make sure it's available in dask threads
    if pairwise_reg_func == registration_ITKElastix:
        try:
            global itk
            import itk
        except ImportError:
            raise ImportError(
                "Please install the itk-elastix package to use ITKElastix for registration.\n"
                "E.g. using pip:\n"
                "- `pip install multiview-stitcher[itk-elastix]` or\n"
                "- `pip install itk-elastix`"
            ) from None

    # compute pairwise registrations
    g_reg_computed = compute_pairwise_registrations(
        msims_reg,
        g_reg,
        transform_key=transform_key,
        points_key=points_key,
        prefilter_markers=prefilter_markers,
        registration_binning=registration_binning,
        reg_res_level=reg_res_level,
        overlap_tolerance=overlap_tolerance,
        pairwise_reg_func=pairwise_reg_func,
        pairwise_reg_func_kwargs=pairwise_reg_func_kwargs,
        n_parallel_pairwise_regs=n_parallel_pairwise_regs,
    )

    # optionally filter obtained pairwise registrations by quality
    if post_registration_do_quality_filter:
        # filter edges by quality
        g_reg_computed = mv_graph.filter_edges(
            g_reg_computed,
            threshold=post_registration_quality_threshold,
            weight_key="quality",
        )

    # resolve global registration parameters from pairwise registrations
    params_dict, groupwise_resolution_info_dict = groupwise_resolution(
        g_reg_computed,
        method=groupwise_resolution_method,
        **groupwise_resolution_kwargs,
    )
    if logger.isEnabledFor(logging.DEBUG):
        for view_index, params_for_view in sorted(params_dict.items()):
            logger.debug(
                "Groupwise registration transform: view=%s, method=%s, affine=\n%s",
                view_index,
                groupwise_resolution_method,
                _format_array_for_log(params_for_view),
            )

    params = [
        params_dict[iview] for iview in sorted(g_reg_computed.nodes())
    ]

    # optionally write registration result back to the input msims
    # under a new transform key
    if new_transform_key is not None:
        for imsim, msim in enumerate(msims):
            msi_utils.set_affine_transform(
                msim,
                params[imsim],
                transform_key=new_transform_key,
                base_transform_key=transform_key,
            )

    # optionally plot registration summaries
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
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Computed pairwise registration: edge=%s, quality=%s, "
                "transform_shape=%s, transform=\n%s\nbbox=%s",
                pair,
                _format_array_for_log(params[i]["quality"]),
                np.asarray(params[i]["transform"]).shape,
                _format_array_for_log(params[i]["transform"]),
                _format_array_for_log(params[i]["bbox"]),
            )

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


def _get_elastix_probe_points(ndim, samples_per_axis=5, cube_extent=10.0):
    axes = [np.linspace(0.0, cube_extent, samples_per_axis) for _ in range(ndim)]
    grid = np.meshgrid(*axes, indexing="ij")
    return np.stack(grid, axis=-1).reshape(-1, ndim)


def _to_itk_spatial_order(values):
    return tuple(float(value) for value in values[::-1])


def _points_to_itk_spatial_order(points):
    return np.asarray(points, dtype=float)[:, ::-1]


def _points_from_itk_spatial_order(points):
    return np.asarray(points, dtype=float)[:, ::-1]


def _write_elastix_point_set_file(path, points):
    with open(path, "w", encoding="ascii") as point_file:
        point_file.write("point\n")
        point_file.write(f"{len(points)}\n")
        for point in points:
            point_file.write(
                " ".join(str(float(value)) for value in point) + "\n"
            )


def _parse_elastix_output_points(path):
    transformed_points = []
    with open(path, encoding="ascii") as output_file:
        for line in output_file:
            if "OutputPoint = [" not in line:
                continue
            point_str = line.split("OutputPoint = [", 1)[1].split("]", 1)[0]
            transformed_points.append([float(value) for value in point_str.split()])

    return _points_from_itk_spatial_order(transformed_points)


def _get_itk_image_from_data(data, *, origin, spacing):
    image = itk.image_view_from_array(np.asarray(data, dtype=np.float32))
    image.SetOrigin(_to_itk_spatial_order(origin))
    image.SetSpacing(_to_itk_spatial_order(spacing))
    return image


def _get_elastix_affine_parameter_map(initial_affine, ndim):
    affine = np.asarray(initial_affine, dtype=float)
    itk_matrix = affine[:ndim, :ndim][::-1, ::-1]
    center_of_rotation = np.zeros(ndim, dtype=float)
    itk_offset = (
        affine[:ndim, ndim] + (affine[:ndim, :ndim] - np.eye(ndim)) @ center_of_rotation
    )[::-1]

    return {
        "Transform": ["AffineTransform"],
        "NumberOfParameters": [str(ndim * (ndim + 1))],
        "TransformParameters": [
            str(value)
            for value in np.concatenate(
                [itk_matrix.reshape(-1), itk_offset]
            )
        ],
        "CenterOfRotationPoint": [
            str(value) for value in center_of_rotation[::-1]
        ],
        "InitialTransformParameterFileName": ["NoInitialTransform"],
        "HowToCombineTransforms": ["Compose"],
        "FixedImageDimension": [str(ndim)],
        "MovingImageDimension": [str(ndim)],
        "FixedInternalImagePixelType": ["float"],
        "MovingInternalImagePixelType": ["float"],
        "Size": ["1"] * ndim,
        "Index": ["0"] * ndim,
        "Spacing": ["1"] * ndim,
        "Origin": ["0"] * ndim,
        "Direction": [str(value) for value in np.eye(ndim).reshape(-1)],
        "UseDirectionCosines": ["true"],
        "ResampleInterpolator": ["FinalBSplineInterpolator"],
        "Resampler": ["DefaultResampler"],
        "DefaultPixelValue": ["0"],
        "CompressResultImage": ["false"],
        "FinalBSplineInterpolationOrder": ["3"],
        "ResultImagePixelType": ["float32"],
        "ResultImageFormat": ["nii"],
    }


def _get_elastix_parameter_map(
    transform_type,
    number_of_resolutions=2,
    number_of_iterations=None,
    metric=None,
    write_result_image=False,
):
    transform_type_map = {
        "translation": ("translation", "TranslationTransform"),
        "rigid": ("rigid", "EulerTransform"),
        "similarity": ("rigid", "SimilarityTransform"),
        "affine": ("affine", "AffineTransform"),
    }

    normalized_transform_type = transform_type.lower()
    if normalized_transform_type not in transform_type_map:
        raise ValueError(
            f"Unsupported elastix transform type: {transform_type}"
        )

    default_map_name, elastix_transform_name = transform_type_map[
        normalized_transform_type
    ]
    parameter_map = itk.ParameterObject.GetDefaultParameterMap(
        default_map_name, number_of_resolutions
    )
    parameter_map["Transform"] = [elastix_transform_name]
    parameter_map["AutomaticTransformInitialization"] = ["false"]
    parameter_map["WriteResultImage"] = [str(write_result_image).lower()]

    if number_of_iterations is not None:
        parameter_map["MaximumNumberOfIterations"] = [
            str(number_of_iterations)
        ] * number_of_resolutions

    if metric is not None:
        parameter_map["Metric"] = [metric]

    return parameter_map


def _write_initial_elastix_transform(
    path,
    *,
    initial_affine,
    ndim,
):
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterMap(
        _get_elastix_affine_parameter_map(initial_affine, ndim)
    )
    parameter_object.WriteParameterFile(path)


def _get_affine_from_elastix_transform_parameter_object(
    transform_parameter_object,
    *,
    moving_image,
    ndim,
):
    fixed_points = _get_elastix_probe_points(ndim)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_points_path = os.path.join(tmpdir, "input_points.txt")
        output_dir = os.path.join(tmpdir, "transformix_output")
        os.makedirs(output_dir)

        _write_elastix_point_set_file(
            input_points_path,
            _points_to_itk_spatial_order(fixed_points),
        )

        itk.transformix_filter(
            moving_image=moving_image,
            transform_parameter_object=transform_parameter_object,
            output_directory=output_dir,
            fixed_point_set_file_name=input_points_path,
            log_to_console=False,
        )

        moving_points = _parse_elastix_output_points(
            os.path.join(output_dir, "outputpoints.txt")
        )

    fixed_to_moving = AffineTransform()
    fixed_to_moving.estimate(fixed_points, moving_points)

    return param_utils.affine_to_xaffine(fixed_to_moving.params)


def registration_ITKElastix(
    fixed_data,
    moving_data,
    *,
    fixed_origin,
    moving_origin,
    fixed_spacing,
    moving_spacing,
    initial_affine,
    transform_types=None,
    **elastix_registration_kwargs,
):
    """
    Use ITKElastix to perform registration between two spatial images.

    Parameters
    ----------
    transform_types : list of str, optional
        Sequence of transform types to apply in successive stages.
        Supported values: 'Translation', 'Rigid', 'Similarity', 'Affine'.
        By default ['Translation', 'Rigid', 'Similarity'].
    **elastix_registration_kwargs
        Additional keyword arguments. The following are handled explicitly
        and applied to the elastix parameter map for each stage:

        number_of_resolutions : int, optional
            Number of resolution levels in the multi-resolution scheme,
            by default 2.
        number_of_iterations : int, optional
            Maximum number of optimizer iterations per resolution level.
            If None, the elastix default for the chosen transform type is used.
        metric : str, optional
            Similarity metric used by elastix. If None, the elastix default
            for the chosen transform type is used. Common values:

            - 'AdvancedMattesMutualInformation' (default for most transforms)
            - 'AdvancedMeanSquares'
            - 'AdvancedNormalizedCorrelation'
            - 'NormalizedMutualInformation'

        Remaining kwargs are forwarded to ``itk.elastix_registration_method``
        (e.g. ``log_to_console=True``).
    """

    try:
        global itk
        import itk
    except ImportError:
        raise ImportError(
            "Please install the itk-elastix package to use ITKElastix for registration.\n"
            "E.g. using pip:\n"
            "- `pip install multiview-stitcher[itk-elastix]` or\n"
            "- `pip install itk-elastix`"
        ) from None

    if transform_types is None:
        transform_types = ["Translation", "Rigid"]

    transform_types = [t.title() for t in transform_types]

    spatial_dims = fixed_data.dims
    ndim = len(spatial_dims)

    fixed_image = _get_itk_image_from_data(
        fixed_data.data,
        origin=[fixed_origin[dim] for dim in spatial_dims],
        spacing=[fixed_spacing[dim] for dim in spatial_dims],
    )
    moving_image = _get_itk_image_from_data(
        moving_data.data,
        origin=[moving_origin[dim] for dim in spatial_dims],
        spacing=[moving_spacing[dim] for dim in spatial_dims],
    )

    number_of_iterations = elastix_registration_kwargs.pop(
        "number_of_iterations", None
    )
    number_of_resolutions = elastix_registration_kwargs.pop(
        "number_of_resolutions", 2
    )
    metric = elastix_registration_kwargs.pop("metric", None)

    default_elastix_registration_kwargs = {
        "log_to_console": False,
    }
    elastix_registration_kwargs = {
        **default_elastix_registration_kwargs,
        **elastix_registration_kwargs,
    }

    # Run one elastix call per transform type, threading the composed affine
    # forward as the initial transform for each successive stage.  This avoids
    # elastix's multi-stage chaining, which breaks when output_directory is not
    # set (IntialTransformParameterFileName becomes '' for stages beyond the
    # first) and can also partially undo the initial transform when chaining.
    with tempfile.TemporaryDirectory() as tmpdir:
        current_affine = initial_affine
        result_image = None

        for i_stage, transform_type in enumerate(transform_types):
            is_last = i_stage == len(transform_types) - 1
            stage_dir = os.path.join(tmpdir, f"stage_{i_stage}")
            os.makedirs(stage_dir)

            initial_transform_path = os.path.join(
                stage_dir, "initial_transform.txt"
            )
            _write_initial_elastix_transform(
                initial_transform_path,
                initial_affine=current_affine,
                ndim=ndim,
            )

            single_stage_po = itk.ParameterObject.New()
            single_stage_po.AddParameterMap(
                _get_elastix_parameter_map(
                    transform_type,
                    number_of_resolutions=number_of_resolutions,
                    number_of_iterations=number_of_iterations,
                    metric=metric,
                    write_result_image=is_last,
                )
            )

            result_image, result_parameter_object = itk.elastix_registration_method(
                fixed_image=fixed_image,
                moving_image=moving_image,
                parameter_object=single_stage_po,
                initial_transform_parameter_file_name=initial_transform_path,
                output_directory=stage_dir,
                **elastix_registration_kwargs,
            )

            current_affine = _get_affine_from_elastix_transform_parameter_object(
                result_parameter_object,
                moving_image=moving_image,
                ndim=ndim,
            )

        affine_matrix = current_affine

    quality = link_quality_metric_func(
        np.asarray(fixed_data.data),
        itk.array_view_from_image(result_image),
    )

    return {
        "affine_matrix": affine_matrix,
        "quality": quality,
    }


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
