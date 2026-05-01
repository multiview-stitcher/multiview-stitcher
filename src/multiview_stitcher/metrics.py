"""
Registration quality metrics for multiview-stitcher.

This module provides tools to assess the quality of image registration by
comparing image content in the overlap regions between adjacent views, after
pre-transforming them according to one or more candidate transform keys.

Main entry point: ``tile_pair_image_metrics``.

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


def _compute_metrics_from_arrays(
        fixed_sim,
        moving_sim,
        metric_funcs,
        intersection_halfspace=None,
    ):
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
    # fixed_np = np.asarray(fixed_arr, dtype=np.float32)
    # moving_np = np.asarray(moving_arr, dtype=np.float32)
    # fixed_sim

    fixed_np = np.asarray(fixed_sim.data, dtype=np.float32)
    moving_np = np.asarray(moving_sim.data, dtype=np.float32)

    if intersection_halfspace is not None:
        # Mask out the half-space of the fixed image that lies outside the
        # overlap region
        mask = mv_graph.get_mask_from_halfspace(
            fixed_sim, intersection_halfspace
        )
        fixed_np[~mask] = np.nan

    return {key: float(func(fixed_np, moving_np)) for key, func in metric_funcs.items()}


def _build_metrics_graph(
    msims,
    sims_t0,
    base_transform_key,
    query_transform_keys,
    max_tolerance,
    bidirectional=False,
):
    """
    Build a directed graph used internally for metric computation.

    Each directed edge ``(fixed_idx, moving_idx)`` stores:

    * ``comparison_bbox`` – bounding box (world coords, base_transform_key)
      for this direction, or ``None`` when no valid overlap exists.
    * ``transforms`` – dict mapping each candidate key to the sampling
      transform ``p_moving`` (ndarray, shape ``(ndim+1, ndim+1)``) that
      maps fixed-intrinsic coordinates to moving-intrinsic coordinates.

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
    bidirectional : bool, optional
        When ``False`` (default) only the directed edge ``(i, j)`` with
        ``i < j`` is built for each undirected adjacency edge.  When
        ``True`` both directions are built.

    Returns
    -------
    nx.DiGraph
    """

    sdims = spatial_image_utils.get_spatial_dims_from_sim(sims_t0[0])

    if max_tolerance is None:
        tol = None
    elif isinstance(max_tolerance, (int, float)):
        tol = -float(max_tolerance)
    else:
        tol = {
            dim: -float(max_tolerance.get(dim, 0.0)) for dim in sdims
        }

    g_adj = mv_graph.build_view_adjacency_graph_from_msims(
        msims, transform_key=base_transform_key, overlap_tolerance=tol
    )

    # log which pairs are considered
    logger.info(
        "Building metrics graph with %s directed edge(s) from %s undirected adjacency edge(s)",
        len(g_adj.edges()) * (2 if bidirectional else 1),
        len(g_adj.edges()),
    )

    g_metrics = nx.DiGraph()
    for node in g_adj.nodes():
        g_metrics.add_node(node)

    for i, j in g_adj.edges():
        directions = [(i, j), (j, i)] if bidirectional else [(min(i, j), max(i, j))]
        for fixed_idx, moving_idx in directions:
            sim_fixed = sims_t0[fixed_idx]
            sim_moving = sims_t0[moving_idx]

            # Negative overlap_tolerance shrinks each sim's bbox before
            # computing the intersection, equivalent to pulling the
            # comparison box inward by max_tolerance on every side.

            overlap_dict = registration._get_overlap_bboxes(
                sim_fixed,
                sim_moving,
                input_transform_key=base_transform_key,
                # output_transform_key=base_transform_key,
                output_transform_key=None,
                overlap_tolerance=tol,
            )
            lowers_phys, uppers_phys = overlap_dict["lowers"], overlap_dict["uppers"]

            lower = np.asarray(lowers_phys[0], dtype=float)
            upper = np.asarray(uppers_phys[0], dtype=float)

            if np.any(lower >= upper):
                comparison_bbox = None
            else:
                comparison_bbox = {"lower": lower, "upper": upper}

            transforms = {}
            for q in query_transform_keys:
                T_fixed_q = (
                    spatial_image_utils.get_affine_from_sim(sim_fixed, q)
                    .squeeze()
                    .data
                )
                T_moving_q = (
                    spatial_image_utils.get_affine_from_sim(sim_moving, q)
                    .squeeze()
                    .data
                )
                transforms[q] = np.linalg.inv(T_moving_q) @ T_fixed_q

            g_metrics.add_edge(
                fixed_idx,
                moving_idx,
                comparison_bbox=comparison_bbox,
                transforms=transforms,
                intersection_halfspace=overlap_dict['intersection'],
                vol=overlap_dict['vol'],
            )

    return g_metrics


def _build_metrics_graph_from_pairs_graph(
    msims,
    sims_t0,
    base_transform_key,
    pairs_graph,
    max_tolerance,
    bidirectional=False,
):
    """
    Build a directed metrics graph from a pre-computed pairwise registration
    graph.

    Mirrors :func:`_build_metrics_graph` but derives pairs and transforms from
    ``pairs_graph`` (e.g. ``g_reg_computed`` from
    :func:`registration.compute_pairwise_registrations`) instead of from the
    msim transform keys.

    The edge ``"transform"`` attribute is expected to map world coordinates of
    the lower-index view (fixed) to world coordinates of the higher-index view
    (moving).  The resulting ``g_metrics`` edges store ``transforms`` in the
    same unified format as :func:`_build_metrics_graph`: a dict with key
    ``"transform"`` mapping to ``p_moving`` (fixed-intrinsic →
    moving-intrinsic ndarray).

    Parameters
    ----------
    msims : list of MultiscaleSpatialImage
    sims_t0 : list of SpatialImage
        First time-point (and first channel) of each view.
    base_transform_key : str
        Transform key used to compute overlap geometry and to convert the
        world-space edge transform to the intrinsic sampling convention.
    pairs_graph : nx.Graph
        Pairwise registration graph.  Each edge ``(i, j)`` with ``i < j``
        must carry a ``"transform"`` attribute (world-space affine,
        lower-index view → higher-index view).
    max_tolerance : float, dict, or None
    bidirectional : bool, optional
        When ``False`` (default) only the edge ``(min_i, max_i)`` direction
        is evaluated.  When ``True`` both directions are evaluated.

    Returns
    -------
    nx.DiGraph
    """
    sdims = spatial_image_utils.get_spatial_dims_from_sim(sims_t0[0])

    if max_tolerance is None:
        tol = None
    elif isinstance(max_tolerance, (int, float)):
        tol = -float(max_tolerance)
    else:
        tol = {
            dim: -float(max_tolerance.get(dim, 0.0)) for dim in sdims
        }

    g_metrics = nx.DiGraph()
    for node in pairs_graph.nodes():
        g_metrics.add_node(node)

    logger.info(
        "Building metrics graph from pairs_graph with %s directed edge(s) "
        "from %s undirected edge(s)",
        pairs_graph.number_of_edges() * (2 if bidirectional else 1),
        pairs_graph.number_of_edges(),
    )

    for i, j in pairs_graph.edges():
        # Canonical direction stored in pairs_graph: lower index = fixed
        fixed_base, moving_base = min(i, j), max(i, j)

        # Extract edge transform: world(fixed_base) → world(moving_base)
        T_edge_raw = pairs_graph.edges[fixed_base, moving_base]["transform"]
        if hasattr(T_edge_raw, "coords") and "t" in T_edge_raw.coords:
            T_edge_raw = T_edge_raw.isel(t=0)
        T_edge = np.asarray(T_edge_raw).squeeze()

        directions = [(fixed_base, moving_base)]
        if bidirectional:
            directions = [(fixed_base, moving_base), (moving_base, fixed_base)]

        for fixed_idx, moving_idx in directions:
            sim_fixed = sims_t0[fixed_idx]
            sim_moving = sims_t0[moving_idx]

            overlap_dict = registration._get_overlap_bboxes(
                sim_fixed,
                sim_moving,
                input_transform_key=base_transform_key,
                output_transform_key=None,
                overlap_tolerance=tol,
            )
            lowers_phys, uppers_phys = overlap_dict["lowers"], overlap_dict["uppers"]

            lower = np.asarray(lowers_phys[0], dtype=float)
            upper = np.asarray(uppers_phys[0], dtype=float)

            if np.any(lower >= upper):
                comparison_bbox = None
            else:
                comparison_bbox = {"lower": lower, "upper": upper}

            T_fixed_base = (
                spatial_image_utils.get_affine_from_sim(sim_fixed, base_transform_key)
                .squeeze()
                .data
            )
            T_moving_base = (
                spatial_image_utils.get_affine_from_sim(sim_moving, base_transform_key)
                .squeeze()
                .data
            )

            # Convert world-space edge transform to fixed-intrinsic → moving-intrinsic.
            # T_edge maps world(fixed_base) → world(moving_base).
            # For the reverse direction, use inv(T_edge).
            if fixed_idx < moving_idx:
                p_moving = np.linalg.inv(T_moving_base) @ T_edge @ T_fixed_base
            else:
                p_moving = np.linalg.inv(T_fixed_base) @ np.linalg.inv(T_edge) @ T_moving_base

            g_metrics.add_edge(
                fixed_idx,
                moving_idx,
                comparison_bbox=comparison_bbox,
                transforms={"transform": p_moving},
                intersection_halfspace=overlap_dict["intersection"],
                vol=overlap_dict["vol"],
            )

    return g_metrics


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def tile_pair_image_metrics(
    msims,
    base_transform_key,
    query_transform_keys=None,
    metric_funcs=None,
    max_tolerance=None,
    spacing=None,
    bidirectional=False,
    metric_channel=None,
    n_parallel_pairs=None,
    input_res_level=None,
    *,
    pairs_graph=None,
):
    """
    Calculate registration quality metrics for a list of views.

    Two modes are supported, selected by providing exactly one of
    ``query_transform_keys`` (Mode 1) or ``pairs_graph`` (Mode 2):

    **Mode 1** – pairs are determined automatically from the spatial overlap
    of the views under ``base_transform_key``; metrics are evaluated under
    each of the supplied ``query_transform_keys``, enabling comparison across
    multiple candidate transforms (e.g. stage vs. registered).

    **Mode 2** – pairs and their transforms are taken directly from a
    pre-computed pairwise registration graph (``pairs_graph``, e.g.
    ``g_reg_computed`` from :func:`registration.compute_pairwise_registrations`).
    Each edge contributes one candidate (its ``"transform"`` attribute).
    Useful for quality assessment and pair filtering between the pairwise
    registration and global resolution steps.

    For each pair the function:

    1. Uses *base_transform_key* to determine the overlap region between
       the two views and computes a *comparison bounding box* (optionally
       shrunk by *max_tolerance* from the overlap boundary).
    2. Projects the comparison bbox into the **fixed image's intrinsic
       (physical) space** via ``inv(T_fixed_base)``.  The fixed image is
       sampled with an identity transform (always the same pixels across
       all query keys).  The moving image is sampled with
       ``inv(T_moving_q) @ T_fixed_q``, i.e. fixed-intrinsic → world via
       the *query* fixed transform, then world → moving-intrinsic.  The
       relative positioning of fixed and moving therefore reflects
       exclusively the query-key transforms, making metrics comparable
       across keys.
    3. Applies every metric function to the pre-transformed image pair.

    Only the first time point (and first channel) of each view is used.

    Parameters
    ----------
    msims : list of MultiscaleSpatialImage
        Input views.
    base_transform_key : str
        Transform key that defines the reference spatial layout.  Used
        to compute overlap regions and to position the fixed image.
    query_transform_keys : str or list of str, optional
        *Mode 1* — one or more transform keys to evaluate.  Each key must
        exist in every input view.  Mutually exclusive with ``pairs_graph``.
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
        overlap boundary. This guarantees that the comparison bbox
        remains valid for any query transform that deviates from the base
        by at most *max_tolerance* physical units. Pixels that are included
        in the axis-aligned comparison bbox but lie outside of the
        shrunk overlap halfspace intersection are set to NaN before metric evaluation.
        A float value is applied uniformly across all spatial dimensions;
        a dict maps spatial dim names to per-dimension values.
        ``None`` means no shrinkage.
    spacing : dict or None, optional
        Spacing at which images are pretransformed before metric
        evaluation.  A dict maps spatial dim names to per-dimension
        values.  ``None`` (default) uses the finest spacing of the fixed
        image for each pair, preserving the full resolution of the
        reference view.
    bidirectional : bool, optional
        When ``False`` (default) only one directed edge per adjacent pair is
        built, with the lower view index as fixed and the higher as moving.
        This halves the computation cost.  When ``True`` both directions
        ``(i → j)`` and ``(j → i)`` are evaluated independently.
    metric_channel : scalar or None, optional
        Channel coordinate value to use when selecting the channel for metric
        computation.  When ``None`` (default) the channel at index 0 is used.
        Has no effect for views without a ``"c"`` dimension.
    n_parallel_pairs : int or None, optional
        Maximum number of directed pairs to compute in parallel.  When
        ``None`` (default) all pairs are computed in a single :func:`dask.compute`
        call.  For 3D data this defaults to ``1`` to limit memory usage.
        Setting this to a small integer batches the computation, reducing peak
        memory at the cost of reduced parallelism.
    input_res_level : int or None, optional
        Resolution level index used to select the image scale for metric
        computation.  ``0`` is the finest level (``"scale0"``), ``1`` is
        ``"scale1"``, etc.

        * When ``None`` and *spacing* is also ``None``: defaults to ``0``
          (finest resolution).
        * When ``None`` and *spacing* is provided: the coarsest level whose
          actual spacing is still ≤ *spacing*, selected independently for
          each pair, based on the fixed image.
    pairs_graph : nx.Graph, optional
        *Mode 2* — a pre-computed pairwise registration graph (e.g.
        ``g_reg_computed`` returned by
        :func:`registration.compute_pairwise_registrations`).  Each edge
        must carry a ``"transform"`` attribute (the world-space pairwise
        affine, lower-index view → higher-index view).  The edges define
        which pairs are evaluated; each edge contributes a single candidate
        transform.  Mutually exclusive with ``query_transform_keys``.
        The output ``"pairs"`` dict uses ``"transform"`` as the candidate
        key.

    Returns
    -------
    dict with keys:

    * ``"pairs"`` – :class:`dict` mapping directional-pair tuples
      ``(fixed_idx, moving_idx)`` to dicts of the form
      ``{candidate_key: {metric_key: float}}``, where ``candidate_key``
      is a query transform key name (Mode 1) or ``"transform"`` (Mode 2).
    * ``"summary"`` – :class:`dict` mapping *query_transform_key* to
      ``{metric_key: float}`` where each value is the **overlap-volume-weighted
      mean** across all directional pairs.  The weight for each pair is the
      physical volume of the overlap region (as returned by
      :func:`mv_graph.get_overlap_between_pair_of_stack_props`).  Pairs whose
      metric value is NaN are excluded from both the numerator and denominator.
    """
    if (query_transform_keys is None) == (pairs_graph is None):
        raise ValueError(
            "Exactly one of 'query_transform_keys' or 'pairs_graph' must be provided."
        )

    if metric_funcs is None:
        metric_funcs = {"ncc": normalized_cross_correlation}

    if query_transform_keys is not None:
        if isinstance(query_transform_keys, str):
            query_transform_keys = [query_transform_keys]
        candidate_keys = query_transform_keys
    else:
        candidate_keys = ["transform"]

    # Resolve input_res_level when not explicitly set.
    # Per-pair selection (input_res_level stays None) only happens when
    # spacing is provided; otherwise we fall back to the finest level.
    per_pair_res_level = False
    if input_res_level is None:
        if spacing is None:
            input_res_level = 0
        else:
            per_pair_res_level = True

    # Build sims_t0 for graph construction (overlap / adjacency).  When
    # the resolution level is fixed we use that level directly; for the
    # per-pair case we use scale0 here (transforms always come from scale0).
    graph_scale_key = "scale0" if per_pair_res_level else f"scale{input_res_level}"
    sims = [msi_utils.get_sim_from_msim(msim, scale=graph_scale_key) for msim in msims]
    spatial_dims = spatial_image_utils.get_spatial_dims_from_sim(sims[0])
    ndim = len(spatial_dims)

    # Select first time-point and chosen channel from each sim
    sims_t0 = []
    for sim in sims:
        sel = {}
        if "t" in sim.dims:
            sel["t"] = sim.coords["t"].values[0]
        if "c" in sim.dims:
            if metric_channel is None:
                sel["c"] = sim.coords["c"].values[0]
            else:
                sel["c"] = metric_channel
        if sel:
            sim = spatial_image_utils.sim_sel_coords(sim, sel)
        sims_t0.append(sim)

    # spacing is a dict or None; kept as-is for per-pair use inside the loop.
    spacing_global = spacing

    # Build directed metrics graph
    if query_transform_keys is not None:
        g_metrics = _build_metrics_graph(
            msims,
            sims_t0,
            base_transform_key,
            query_transform_keys,
            max_tolerance,
            bidirectional=bidirectional,
        )
    else:
        g_metrics = _build_metrics_graph_from_pairs_graph(
            msims,
            sims_t0,
            base_transform_key,
            pairs_graph,
            max_tolerance,
            bidirectional=bidirectional,
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
        intersection_halfspace = edge_data["intersection_halfspace"]
        vol=edge_data["vol"]

        # expand halfspace slightly to make sure it includes the boundary of the intersection
        fixed_spacing = spatial_image_utils.get_spacing_from_sim(sims_t0[fixed_idx], asarray=True)
        tol = 1e-3 * np.min(fixed_spacing)
        intersection_halfspace = mv_graph.expand_halfspace(intersection_halfspace, distance=tol)

        if comparison_bbox is None:
            logger.warning(
                "Empty comparison bbox for directed pair (%s → %s), "
                "all metrics will be NaN.",
                fixed_idx,
                moving_idx,
            )
            metric_delayed[(fixed_idx, moving_idx)] = {
                q: {k: np.nan for k in metric_funcs} for q in candidate_keys
            }
            continue

        # Select the sims for metric computation at the appropriate resolution.
        if per_pair_res_level:
            pair_res_level = msi_utils.get_res_level_from_spacing(
                msims[fixed_idx], spacing
            )
            pair_scale_key = f"scale{pair_res_level}"
            def _get_sim_t0(msim, scale_key):
                sim = msi_utils.get_sim_from_msim(msim, scale=scale_key)
                sel = {}
                if "t" in sim.dims:
                    sel["t"] = sim.coords["t"].values[0]
                if "c" in sim.dims:
                    if metric_channel is None:
                        sel["c"] = sim.coords["c"].values[0]
                    else:
                        sel["c"] = metric_channel
                if sel:
                    sim = spatial_image_utils.sim_sel_coords(sim, sel)
                return sim
            sim_fixed = _get_sim_t0(msims[fixed_idx], pair_scale_key)
            sim_moving = _get_sim_t0(msims[moving_idx], pair_scale_key)
        else:
            sim_fixed = sims_t0[fixed_idx]
            sim_moving = sims_t0[moving_idx]

        lower_intrinsic = comparison_bbox["lower"]
        upper_intrinsic = comparison_bbox["upper"]

        # Resolve per-pair spacing: use caller-supplied dict or fall back to
        # the spacing of the resolution level corresponding to input_res_level
        # for the fixed image (sim_fixed already comes from that level).
        if spacing_global is not None:
            spacing_d = spacing_global
        else:
            spacing_d = spatial_image_utils.get_spacing_from_sim(sim_fixed)

        shape = {
            dim: max(
                1,
                int(np.floor(
                    (upper_intrinsic[idim] - lower_intrinsic[idim]) / spacing_d[dim] + 1
                )),
            )
            for idim, dim in enumerate(spatial_dims)
        }

        # output_sp is in fixed-image intrinsic (physical) space
        output_sp = {
            "origin": {
                dim: float(lower_intrinsic[idim]) for idim, dim in enumerate(spatial_dims)
            },
            "spacing": {
                dim: float(spacing_d[dim]) for dim in spatial_dims
            },
            "shape": {
                dim: int(shape[dim]) for dim in spatial_dims
            },
        }

        # Fixed image: identity transform — output space IS fixed-intrinsic
        # space, so the fixed image is read out directly with no resampling.
        # This is computed once per directed edge and shared across all query
        # keys, guaranteeing identical fixed-image content for every comparison.
        sim_fixed_t = transformation.transform_sim(
            sim_fixed.astype(np.float32),
            p=np.eye(ndim + 1),
            output_stack_properties=output_sp,
            mode="constant",
            cval=np.nan,
            order=1,
        )

        # Moving image: for each candidate key retrieve the pre-computed
        # p_moving (fixed-intrinsic → moving-intrinsic) stored on the edge.
        metric_delayed[(fixed_idx, moving_idx)] = {}

        for q in candidate_keys:
            p_moving = transforms[q]

            sim_moving_t = transformation.transform_sim(
                sim_moving.astype(np.float32),
                p=p_moving,
                output_stack_properties=output_sp,
                mode="constant",
                cval=np.nan,
            )

            metric_d = delayed(_compute_metrics_from_arrays)(
                sim_fixed_t,
                sim_moving_t,
                metric_funcs,
                intersection_halfspace.halfspaces,
            )
            metric_delayed[(fixed_idx, moving_idx)][q] = metric_d

    # Compute all pairs and all query keys in parallel,
    # optionally batched to limit peak memory usage.

    if n_parallel_pairs is None and ndim == 3:
        n_parallel_pairs = 1
        logger.info("Setting n_parallel_pairs to 1 for 3D data")

    if n_parallel_pairs is None:
        logger.info("Computing metrics for all pairs in parallel")
        computed = compute(metric_delayed)[0]
    else:
        logger.info("Computing metrics for %s pair(s) in parallel", n_parallel_pairs)
        computed = {}
        all_pairs = list(metric_delayed.keys())
        for i in range(0, len(all_pairs), n_parallel_pairs):
            batch_pairs = all_pairs[i : i + n_parallel_pairs]
            batch = {p: metric_delayed[p] for p in batch_pairs}
            computed.update(compute(batch)[0])

    # Store computed metrics back on the graph edges
    for fixed_idx, moving_idx in g_metrics.edges():
        g_metrics.edges[(fixed_idx, moving_idx)]["metrics"] = computed[
            (fixed_idx, moving_idx)
        ]

    # -----------------------------------------------------------------------
    # Summarise: overlap-volume-weighted mean over all directed pairs,
    # per query key, per metric key
    # -----------------------------------------------------------------------
    summary = {}
    for q in candidate_keys:
        summary[q] = {}
        for metric_key in metric_funcs:
            values_and_weights = [
                (
                    float(computed[(fi, mi)][q].get(metric_key, np.nan)),
                    float(g_metrics.edges[(fi, mi)]["vol"]),
                )
                for fi, mi in g_metrics.edges()
            ]
            valid = [(v, w) for v, w in values_and_weights if not np.isnan(v)]
            if valid:
                vals, weights = zip(*valid)
                total_w = sum(weights)
                summary[q][metric_key] = (
                    float(sum(v * w for v, w in zip(vals, weights)) / total_w)
                    if total_w > 0
                    else np.nan
                )
            else:
                summary[q][metric_key] = np.nan

    return {
        "pairs": {
            (fi, mi): {
                q: computed[(fi, mi)][q] for q in candidate_keys
            }
            for fi, mi in g_metrics.edges()
        },
        "bboxes": {
            (fi, mi): g_metrics.edges[(fi, mi)]["comparison_bbox"]
            for fi, mi in g_metrics.edges()
        },
        "summary": summary,
    }
