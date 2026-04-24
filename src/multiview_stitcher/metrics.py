"""
Registration quality metrics for multiview-stitcher.

This module provides tools to assess the quality of image registration by
comparing image content in the overlap regions between adjacent views, after
pre-transforming them according to one or more candidate transform keys.

Main entry point
----------------
``calc_reg_metrics`` computes pairwise image similarity metrics for all
adjacent view pairs under a set of query transform keys, returning both
per-pair values and summarised statistics.

Built-in metric function
------------------------
``normalized_cross_correlation`` – normalised cross-correlation (NCC) between
two images that may contain NaN values in non-overlapping areas.

Additional metric functions, such as those from :mod:`skimage.metrics`, can be
passed through the ``metric_funcs`` argument as long as they conform to the
signature ``func(im1: np.ndarray, im2: np.ndarray) -> float``.
"""

import logging

import networkx as nx
import numpy as np
import xarray as xr
from dask import compute, delayed

from multiview_stitcher import (
    msi_utils,
    mv_graph,
    registration,
    spatial_image_utils,
    transformation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Built-in metric functions
# ---------------------------------------------------------------------------


def normalized_cross_correlation(im1, im2):
    """
    Compute the normalised cross-correlation (NCC) between two images.

    NaN pixels present in either image are excluded from the computation.

    Parameters
    ----------
    im1 : array-like
        First image (fixed). Arbitrary shape; must match ``im2``.
    im2 : array-like
        Second image (moving). Arbitrary shape; must match ``im1``.

    Returns
    -------
    float
        NCC value in the range [-1, 1].  Returns ``np.nan`` when fewer than
        two overlapping (non-NaN) pixels are available or when either image
        is constant.
    """
    a = np.asarray(im1, dtype=np.float64)
    b = np.asarray(im2, dtype=np.float64)

    mask = ~(np.isnan(a) | np.isnan(b))
    if np.sum(mask) < 2:
        return np.nan

    a = a[mask]
    b = b[mask]

    a_c = a - a.mean()
    b_c = b - b.mean()

    denom = np.sqrt(np.sum(a_c**2) * np.sum(b_c**2))
    if denom < 1e-10:
        return np.nan

    return float(np.dot(a_c, b_c) / denom)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_comparison_bbox(
    sim_fixed,
    sim_moving,
    base_transform_key,
    max_tolerance,
):
    """
    Return the comparison bounding box in base_transform_key world coordinates.

    The box is the overlap region of the two views (computed from
    *base_transform_key*), optionally shrunk inward by *max_tolerance* on
    every side so that the box remains within the moving image even when
    the query transform deviates from the base by up to *max_tolerance*
    physical units.

    Parameters
    ----------
    sim_fixed, sim_moving : SpatialImage
        The two views (first time point, no non-spatial dims).
    base_transform_key : str
        Transform key that defines the reference spatial layout.
    max_tolerance : float, dict, or None
        Amount by which to shrink the overlap box on each side (physical
        units).  A ``float`` is applied uniformly; a ``dict`` maps spatial
        dimension names to per-dimension values.  ``None`` means no
        shrinkage.

    Returns
    -------
    dict
        ``{"lower": np.ndarray, "upper": np.ndarray}`` in world coordinates,
        or ``None`` when the resulting box is empty.
    """
    ndim = spatial_image_utils.get_ndim_from_sim(sim_fixed)
    spatial_dims = spatial_image_utils.get_spatial_dims_from_sim(sim_fixed)

    # Overlap in base_transform_key world coords (from the fixed image's
    # perspective, i.e. lowers_phys[0] / uppers_phys[0]).
    lowers_phys, uppers_phys = registration.get_overlap_bboxes(
        sim_fixed,
        sim_moving,
        input_transform_key=base_transform_key,
        output_transform_key=base_transform_key,
    )

    lower = np.asarray(lowers_phys[0], dtype=float)
    upper = np.asarray(uppers_phys[0], dtype=float)

    # Shrink by max_tolerance
    if max_tolerance is not None:
        if isinstance(max_tolerance, (int, float)):
            tol = np.full(ndim, float(max_tolerance))
        else:
            tol = np.array(
                [float(max_tolerance.get(dim, 0.0)) for dim in spatial_dims]
            )
        lower = lower + tol
        upper = upper - tol

    if np.any(lower >= upper):
        return None

    return {"lower": lower, "upper": upper}


def _compute_metrics_from_arrays(fixed_arr, moving_arr, metric_funcs):
    """
    Apply a dict of metric functions to two pre-transformed image arrays.

    Parameters
    ----------
    fixed_arr : array-like
        Fixed image data.
    moving_arr : array-like
        Moving image data.
    metric_funcs : dict[str, callable]
        Maps metric keys to functions with signature
        ``func(im1: np.ndarray, im2: np.ndarray) -> float``.

    Returns
    -------
    dict[str, float]
    """
    fixed_np = np.asarray(fixed_arr, dtype=np.float64)
    moving_np = np.asarray(moving_arr, dtype=np.float64)
    return {key: func(fixed_np, moving_np) for key, func in metric_funcs.items()}


def _build_metrics_graph(
    msims,
    sims_t0,
    base_transform_key,
    query_transform_keys,
    max_tolerance,
):
    """
    Build a directed graph used internally for metric computation.

    Each directed edge ``(fixed_idx, moving_idx)`` stores:

    * ``comparison_bbox`` – bounding box (world coords, base_transform_key)
      for this direction, or ``None`` when no valid overlap exists.
    * ``transforms`` – dict mapping each query_transform_key to the
      corresponding moving-image affine matrix (ndarray, shape
      ``(ndim+1, ndim+1)``).

    The graph mirrors the ``g_reg_computed`` convention used in
    :func:`registration.register`, extended to be directed and to hold
    per-query-key transforms instead of a single registration result.

    Parameters
    ----------
    msims : list of MultiscaleSpatialImage
    sims_t0 : list of SpatialImage
        First time-point (and first channel) of each view.
    base_transform_key : str
    query_transform_keys : list of str
    max_tolerance : float, dict, or None

    Returns
    -------
    nx.DiGraph
    """
    g_adj = mv_graph.build_view_adjacency_graph_from_msims(
        msims, transform_key=base_transform_key
    )

    g_metrics = nx.DiGraph()
    for node in g_adj.nodes():
        g_metrics.add_node(node)

    for i, j in g_adj.edges():
        for fixed_idx, moving_idx in [(i, j), (j, i)]:
            sim_fixed = sims_t0[fixed_idx]
            sim_moving = sims_t0[moving_idx]

            comparison_bbox = _get_comparison_bbox(
                sim_fixed,
                sim_moving,
                base_transform_key,
                max_tolerance,
            )

            transforms = {
                q: spatial_image_utils.get_affine_from_sim(
                    sim_moving, q
                )
                .squeeze()
                .data
                for q in query_transform_keys
            }

            g_metrics.add_edge(
                fixed_idx,
                moving_idx,
                comparison_bbox=comparison_bbox,
                transforms=transforms,
            )

    return g_metrics


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def calc_reg_metrics(
    msims,
    base_transform_key,
    query_transform_keys,
    metric_funcs=None,
    max_tolerance=None,
    spacing=None,
):
    """
    Calculate registration quality metrics for a list of views.

    For each pair of adjacent views and for both possible fixed/moving
    directions, the function:

    1. Uses *base_transform_key* to determine the overlap region between
       the two views and computes a *comparison bounding box* (optionally
       shrunk by *max_tolerance* from the overlap boundary).
    2. Pre-transforms the fixed image into the comparison bbox using
       *base_transform_key* and the moving image using each
       *query_transform_key*.
    3. Applies every metric function to the pre-transformed image pair.

    Only the first time point (and first channel) of each view is used.

    Parameters
    ----------
    msims : list of MultiscaleSpatialImage
        Input views.
    base_transform_key : str
        Transform key that defines the reference spatial layout.  Used
        to compute overlap regions and to position the fixed image.
    query_transform_keys : str or list of str
        One or more transform keys to evaluate.  Each key must exist in
        every input view.
    metric_funcs : dict[str, callable], optional
        Maps arbitrary string keys to metric functions.  Each function
        must have the signature
        ``func(im1: np.ndarray, im2: np.ndarray) -> float``.
        NaN values in the pre-transformed images (outside the image
        domain) can occur and the metric functions should
        handle them gracefully.
        Defaults to ``{"ncc": normalized_cross_correlation}``.

        To pass additional keyword arguments to a metric function, wrap it
        with :func:`functools.partial` before including it in the dict::

            from functools import partial
            from skimage.metrics import structural_similarity

            metric_funcs = {
                "ncc": metrics.normalized_cross_correlation,
                "ssim": partial(structural_similarity, data_range=1.0),
            }
    max_tolerance : float, dict, or None, optional
        Physical distance by
        which the comparison bbox is shrunk on every side relative to the
        overlap boundary.  This guarantees that the comparison bbox
        remains valid for any query transform that deviates from the base
        by at most *max_tolerance* physical units.
        A float is applied uniformly; a dict maps spatial dim names to
        per-dimension values.  ``None`` means no shrinkage.
    spacing : float, dict, or None, optional
        Spacing at which images are pretransformed before metric
        evaluation.  A float is applied uniformly across all spatial
        dimensions; a dict maps spatial dim names to per-dimension
        values.  ``None`` (default) uses the finest spacing of the fixed
        image for each pair, preserving the full resolution of the
        reference view.

    Returns
    -------
    dict with keys:

    * ``"pairs"`` – :class:`dict` mapping directional-pair tuples
      ``(fixed_idx, moving_idx)`` to dicts of the form
      ``{query_transform_key: {metric_key: float}}``.
    * ``"summary"`` – :class:`dict` mapping *query_transform_key* to
      ``{metric_key: float}`` where each value is the mean across all
      directional pairs (NaN pairs are excluded from the mean).
    """
    if metric_funcs is None:
        metric_funcs = {"ncc": normalized_cross_correlation}

    if isinstance(query_transform_keys, str):
        query_transform_keys = [query_transform_keys]

    sims = [msi_utils.get_sim_from_msim(msim) for msim in msims]
    spatial_dims = spatial_image_utils.get_spatial_dims_from_sim(sims[0])
    ndim = len(spatial_dims)

    # Select first time-point and first channel from each sim
    sims_t0 = []
    for sim in sims:
        sel = {}
        if "t" in sim.dims:
            sel["t"] = sim.coords["t"].values[0]
        if "c" in sim.dims:
            sel["c"] = sim.coords["c"].values[0]
        if sel:
            sim = spatial_image_utils.sim_sel_coords(sim, sel)
        sims_t0.append(sim)

    # Pre-compute a fixed spacing array when the caller supplies one explicitly.
    # When spacing=None the per-pair spacing is resolved inside the loop.
    if spacing is not None:
        if isinstance(spacing, (int, float)):
            spacing_arr_global = np.full(ndim, float(spacing))
        else:
            spacing_arr_global = np.array(
                [float(spacing[dim]) for dim in spatial_dims]
            )
    else:
        spacing_arr_global = None

    # Build directed metrics graph (both directions per undirected edge)
    g_metrics = _build_metrics_graph(
        msims,
        sims_t0,
        base_transform_key,
        query_transform_keys,
        max_tolerance,
    )

    # -----------------------------------------------------------------------
    # Build delayed metric computations for every directed edge and every
    # query_transform_key.  The fixed-image transformation is computed
    # once per directed edge and reused for all query keys.
    # -----------------------------------------------------------------------
    metric_delayed = {}

    for fixed_idx, moving_idx in list(g_metrics.edges()):
        edge_data = g_metrics.edges[(fixed_idx, moving_idx)]
        comparison_bbox = edge_data["comparison_bbox"]
        transforms = edge_data["transforms"]

        if comparison_bbox is None:
            logger.warning(
                "Empty comparison bbox for directed pair (%s → %s), "
                "all metrics will be NaN.",
                fixed_idx,
                moving_idx,
            )
            metric_delayed[(fixed_idx, moving_idx)] = {
                q: {k: np.nan for k in metric_funcs} for q in query_transform_keys
            }
            continue

        sim_fixed = sims_t0[fixed_idx]
        sim_moving = sims_t0[moving_idx]

        # Resolve per-pair spacing: finest spacing of the fixed image
        # when the caller did not supply an explicit value.
        if spacing_arr_global is not None:
            spacing_arr = spacing_arr_global
        else:
            spacing_arr = spatial_image_utils.get_spacing_from_sim(
                sim_fixed, asarray=True
            )

        lower = comparison_bbox["lower"]
        upper = comparison_bbox["upper"]
        shape = np.maximum(
            1, np.floor((upper - lower) / spacing_arr + 1).astype(int)
        )

        output_sp = {
            "origin": {
                dim: float(lower[idim]) for idim, dim in enumerate(spatial_dims)
            },
            "spacing": {
                dim: float(spacing_arr[idim]) for idim, dim in enumerate(spatial_dims)
            },
            "shape": {
                dim: int(shape[idim]) for idim, dim in enumerate(spatial_dims)
            },
        }

        # Transform the fixed image into the comparison bbox
        T_fixed_base = (
            spatial_image_utils.get_affine_from_sim(sim_fixed, base_transform_key)
            .squeeze()
            .data
        )
        p_fixed = np.linalg.inv(T_fixed_base)

        sim_fixed_t = transformation.transform_sim(
            sim_fixed.astype(np.float32),
            p=p_fixed,
            output_stack_properties=output_sp,
            mode="constant",
            cval=np.nan,
        )

        # For each query key, transform the moving image and schedule metrics
        metric_delayed[(fixed_idx, moving_idx)] = {}

        for q in query_transform_keys:
            T_moving_q = transforms[q]
            p_moving = np.linalg.inv(T_moving_q)

            sim_moving_t = transformation.transform_sim(
                sim_moving.astype(np.float32),
                p=p_moving,
                output_stack_properties=output_sp,
                mode="constant",
                cval=np.nan,
            )

            metric_d = delayed(_compute_metrics_from_arrays)(
                sim_fixed_t.data,
                sim_moving_t.data,
                metric_funcs,
            )
            metric_delayed[(fixed_idx, moving_idx)][q] = metric_d

    # Compute all pairs and all query keys in parallel
    computed = compute(metric_delayed)[0]

    # Store computed metrics back on the graph edges
    for fixed_idx, moving_idx in g_metrics.edges():
        g_metrics.edges[(fixed_idx, moving_idx)]["metrics"] = computed[
            (fixed_idx, moving_idx)
        ]

    # -----------------------------------------------------------------------
    # Summarise: mean over all directed pairs, per query key, per metric key
    # -----------------------------------------------------------------------
    summary = {}
    for q in query_transform_keys:
        summary[q] = {}
        for metric_key in metric_funcs:
            values = [
                float(computed[(fi, mi)][q].get(metric_key, np.nan))
                for fi, mi in g_metrics.edges()
            ]
            valid = [v for v in values if not np.isnan(v)]
            summary[q][metric_key] = float(np.mean(valid)) if valid else np.nan

    return {
        "pairs": {
            (fi, mi): {
                q: computed[(fi, mi)][q] for q in query_transform_keys
            }
            for fi, mi in g_metrics.edges()
        },
        "summary": summary,
    }
