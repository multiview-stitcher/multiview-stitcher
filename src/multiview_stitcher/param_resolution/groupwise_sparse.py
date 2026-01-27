import numpy as np
import pandas as pd
import xarray as xr
import networkx as nx
from scipy import sparse
from scipy.sparse.linalg import lsqr
from scipy.spatial.transform import Rotation

from multiview_stitcher import mv_graph, param_utils
from .utils import compute_edge_residuals, get_graph_ndim


def _get_edge_weight(edge_data, weight_mode):
    """Compute scalar edge weight from quality/overlap metadata."""
    quality = edge_data.get("quality", 1.0)
    if isinstance(quality, xr.DataArray):
        quality = quality.data
    quality = float(np.mean(quality))

    overlap = edge_data.get("overlap", 1.0)
    if isinstance(overlap, xr.DataArray):
        overlap = overlap.data
    overlap = float(np.mean(overlap))

    if weight_mode == "quality_overlap":
        weight = quality * overlap
    elif weight_mode == "quality":
        weight = quality
    elif weight_mode == "overlap":
        weight = overlap
    elif weight_mode == "uniform":
        weight = 1.0
    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")

    if not np.isfinite(weight) or weight < 0:
        weight = 0.0
    return weight


def _closest_rotation_and_scale(linear, ndim):
    """Return closest rotation (polar) and uniform scale for a linear map."""
    u, s, vt = np.linalg.svd(linear)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    scale = float(np.mean(s))
    if scale <= 0:
        scale = 1.0
    return r, scale


def _rotation_to_vector(rmat, ndim):
    """Convert a rotation matrix to a minimal rotation vector."""
    if ndim == 2:
        theta = np.arctan2(rmat[1, 0], rmat[0, 0])
        return np.array([theta], dtype=float)
    return Rotation.from_matrix(rmat).as_rotvec()


def _vector_to_rotation(rotvec, ndim):
    """Convert a minimal rotation vector to a rotation matrix."""
    if ndim == 2:
        theta = float(rotvec[0])
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -s], [s, c]], dtype=float)
    return Rotation.from_rotvec(rotvec).as_matrix()


def _get_bbox_center(edge_data, ndim):
    # Use overlap bbox center to anchor translation constraints in physical units.
    bbox = edge_data.get("bbox")
    if bbox is None:
        return np.zeros(ndim, dtype=float)
    if isinstance(bbox, xr.DataArray):
        bbox = bbox.data
    bbox = np.asarray(bbox, dtype=float)
    if bbox.shape[0] < 2:
        return np.zeros(ndim, dtype=float)
    return np.mean(bbox[:2], axis=0)


def _build_node_slices(nodes, reference_view, dim):
    # Pack per-node unknowns into a single vector, excluding the reference.
    index = 0
    slices = {}
    for node in nodes:
        if node == reference_view:
            continue
        slices[node] = slice(index, index + dim)
        index += dim
    return slices, index


def _solve_difference_system(
    edges,
    node_slices,
    n_params,
    reference_view,
    dim,
    key,
    prior_lambda,
    **lsqr_kwargs,
):
    # Solve weighted incidence system with optional Tikhonov prior.
    rows = []
    cols = []
    data = []
    b = []
    row_idx = 0

    for edge in edges:
        u = edge["u"]
        v = edge["v"]
        weight = edge["weight"]
        scale = np.sqrt(weight)
        vec = edge[key]

        for k in range(dim):
            b.append(scale * vec[k])
            if u != reference_view:
                rows.append(row_idx)
                cols.append(node_slices[u].start + k)
                data.append(scale)
            if v != reference_view:
                rows.append(row_idx)
                cols.append(node_slices[v].start + k)
                data.append(-scale)
            row_idx += 1

    if prior_lambda > 0 and n_params > 0:
        scale = float(np.sqrt(prior_lambda))
        for node, slc in node_slices.items():
            for k in range(dim):
                b.append(0.0)
                rows.append(row_idx)
                cols.append(slc.start + k)
                data.append(scale)
                row_idx += 1

    if row_idx == 0:
        return np.zeros(n_params, dtype=float)

    mat = sparse.coo_matrix(
        (data, (rows, cols)), shape=(row_idx, n_params)
    ).tocsr()
    b = np.asarray(b, dtype=float)
    result = lsqr(mat, b, **lsqr_kwargs)
    return result[0]


def _unpack_solution(nodes, node_slices, solution, reference_view, dim):
    # Expand the packed solution back to per-node vectors.
    values = {node: np.zeros(dim, dtype=float) for node in nodes}
    for node in nodes:
        if node == reference_view:
            continue
        slc = node_slices[node]
        values[node] = solution[slc]
    return values


def _build_params_from_components(
    nodes,
    translations,
    rotations,
    scales,
    transform,
    ndim,
    x_in=None,
    x_out=None,
):
    # Assemble homogeneous transforms from per-node components.
    params = {}
    for node in nodes:
        if transform == "translation":
            linear = np.eye(ndim)
        elif transform == "rigid":
            linear = _vector_to_rotation(rotations[node], ndim)
        else:
            linear = np.exp(scales[node]) * _vector_to_rotation(
                rotations[node], ndim
            )

        matrix = np.eye(ndim + 1, dtype=float)
        matrix[:ndim, :ndim] = linear
        matrix[:ndim, ndim] = translations[node]
        xparams = param_utils.affine_to_xaffine(matrix)
        if x_in is not None and x_out is not None:
            xparams = xparams.sel(x_in=x_in, x_out=x_out)
        params[node] = xparams
    return params


def _compute_edge_metrics(edges, residuals_by_edge):
    # Combine residuals with edge metadata for reporting/pruning.
    metrics = []
    residuals = []
    for edge in edges:
        edge_key = tuple(sorted((edge["u"], edge["v"])))
        residual = residuals_by_edge.get(edge_key, np.nan)
        metrics.append(
            {
                "u": edge["u"],
                "v": edge["v"],
                "weight": edge["weight"],
                "residual": residual,
            }
        )
        residuals.append(residual)
    return metrics, np.asarray(residuals, dtype=float)


def groupwise_resolution_sparse_two_pass(
    g_reg_component_tp,
    reference_view=None,
    transform="rigid",
    residual_threshold=None,
    mad_k=2.0,
    keep_mst=True,
    weight_mode="quality_overlap",
    prior_lambda=0.0,
    **kwargs,
):
    """
    Fast groupwise resolution via a single global linearization and two-pass
    outlier pruning.

    We solve for per-tile correction transforms in the common stage frame,
    with either translation-only or rigid (SE(2)/SE(3)) parameterization.
    Pairwise registration provides affine transforms A_ij (u -> v) between
    already stage-placed tiles; under perfect stage placement A_ij ≈ I, and
    the true corrections satisfy A_ij ≈ C_j^{-1} C_i. We therefore seek
    corrections that minimize the discrepancy of these relative constraints.

    Model and linearization
    -----------------------
    Each tile i has correction C_i = [R_i, t_i; 0, 1] with R_i ∈ SO(d) and
    t_i ∈ R^d (d=2 or 3). For each edge (i, j), we project the affine A_ij to
    the closest rotation R_ij via polar decomposition. We also form a translation
    measurement using the overlap bounding-box center p (in tile i coordinates):
        d_ij = A_ij(p) - p                       (translation-only)
        d_ij = A_ij(p) - R_ij p                  (rigid)
    This yields a displacement in physical units, consistent with the residual
    metric used for pruning.

    Rotations are parameterized in the Lie algebra as small vectors ω_i
    (scalar angle in 2D, axis-angle in 3D). For small angles we use the
    first-order relation:
        ω_ij ≈ ω_i - ω_j
    These equations are assembled into a sparse weighted incidence system
    (Laplacian least squares) and solved once.

    Given rotations, translations follow from the linear constraints:
        t_i - t_j ≈ R_j d_ij                     (rigid)
        t_i - t_j ≈ d_ij                         (translation-only)
    which again form a sparse weighted incidence system. A small Tikhonov
    prior (prior_lambda) can be added to keep the solution close to the
    stage frame and remove gauge ambiguity.

    Two-pass pruning
    ----------------
    Pass 1 solves on all edges. Residuals are computed as RMS distances between
    corresponding virtual bead pairs (overlap bbox corners) after applying the
    current global transforms, yielding residuals in physical units. Edges with
    residuals above a user-defined absolute threshold or a MAD-based threshold
    (median + mad_k * MAD) are removed; optionally, a minimum spanning tree
    over residuals is retained to preserve connectivity. Pass 2 re-solves on
    the pruned edge set and returns the final transforms and edge metrics.
    """
    if "mode" in kwargs:
        transform = kwargs.pop("mode")
    if "prune_quantile" in kwargs:
        raise TypeError(
            "prune_quantile is not supported; use residual_threshold or mad_k."
        )

    if not g_reg_component_tp.number_of_edges():
        ndim = get_graph_ndim(g_reg_component_tp)
        params = {
            node: param_utils.identity_transform(ndim)
            for node in g_reg_component_tp.nodes
        }
        return params, {"metrics": None, "used_edges": []}

    if transform not in ("translation", "rigid"):
        raise ValueError(f"Unknown transform: {transform}")

    ndim = get_graph_ndim(g_reg_component_tp)
    if ndim not in (2, 3):
        raise ValueError("Only 2D and 3D supported.")

    use_rot = transform == "rigid"
    use_scale = False
    rot_dim = 1 if ndim == 2 else 3

    if reference_view is not None and reference_view in g_reg_component_tp:
        ref_node = reference_view
    else:
        ref_node = mv_graph.get_node_with_maximal_edge_weight_sum_from_graph(
            g_reg_component_tp, weight_key="quality"
        )

    nodes = list(g_reg_component_tp.nodes)

    edges = []
    for edge in g_reg_component_tp.edges:
        sorted_e = tuple(sorted(edge))
        affine = g_reg_component_tp.edges[sorted_e]["transform"]
        if isinstance(affine, xr.DataArray):
            affine = affine.data
        affine = np.asarray(affine, dtype=float)

        bbox_center = _get_bbox_center(g_reg_component_tp.edges[edge], ndim)
        affine_centered = affine

        linear = affine_centered[:ndim, :ndim]
        dvec = affine_centered[:ndim, ndim]

        rot_uv = None
        scale_uv = None
        if use_rot or use_scale:
            rmat, scale = _closest_rotation_and_scale(linear, ndim)
            if use_rot:
                rot_uv = _rotation_to_vector(rmat, ndim)
            if use_scale:
                scale_uv = np.array([float(np.log(scale))], dtype=float)

        if transform == "translation":
            # Measure displacement in physical units at the overlap center.
            dvec = (linear @ bbox_center + dvec) - bbox_center
        elif use_rot:
            # Strip the rotation component to obtain the translational mismatch.
            dvec = (linear @ bbox_center + dvec) - (rmat @ bbox_center)

        edges.append(
            {
                "u": sorted_e[0],
                "v": sorted_e[1],
                "trans": dvec,
                "rot": rot_uv,
                "scale": scale_uv,
                "weight": _get_edge_weight(
                    g_reg_component_tp.edges[edge], weight_mode
                ),
            }
        )

    lsqr_keys = {
        "damp",
        "atol",
        "btol",
        "conlim",
        "iter_lim",
        "show",
        "calc_var",
    }
    lsqr_kwargs = {k: v for k, v in kwargs.items() if k in lsqr_keys}

    def solve_pass(edge_list):
        # Solve rotations, then translations, on the provided edge set.
        if use_rot:
            rot_slices, rot_params = _build_node_slices(
                nodes, ref_node, rot_dim
            )
            rot_solution = _solve_difference_system(
                edge_list,
                rot_slices,
                rot_params,
                ref_node,
                rot_dim,
                "rot",
                prior_lambda,
                **lsqr_kwargs,
            )
            rot_vecs = _unpack_solution(
                nodes, rot_slices, rot_solution, ref_node, rot_dim
            )
        else:
            rot_vecs = {node: np.zeros(rot_dim, dtype=float) for node in nodes}

        scale_vals = {node: 0.0 for node in nodes}

        if transform == "translation":
            rotations = {
                node: np.zeros(rot_dim, dtype=float) for node in nodes
            }
        else:
            rotations = rot_vecs

        translations = {node: np.zeros(ndim, dtype=float) for node in nodes}
        trans_slices, trans_params = _build_node_slices(
            nodes, ref_node, ndim
        )

        rows = []
        cols = []
        data = []
        b = []
        row_idx = 0

        for edge in edge_list:
            u = edge["u"]
            v = edge["v"]
            weight = edge["weight"]
            scale = np.sqrt(weight)
            dvec = edge["trans"]

            if transform == "translation":
                rhs = dvec
            else:
                rmat = _vector_to_rotation(rotations[v], ndim)
                rhs = rmat @ dvec

            for k in range(ndim):
                b.append(scale * rhs[k])
                if u != ref_node:
                    rows.append(row_idx)
                    cols.append(trans_slices[u].start + k)
                    data.append(scale)
                if v != ref_node:
                    rows.append(row_idx)
                    cols.append(trans_slices[v].start + k)
                    data.append(-scale)
                row_idx += 1

        if prior_lambda > 0 and trans_params > 0:
            scale = float(np.sqrt(prior_lambda))
            for node, slc in trans_slices.items():
                for k in range(ndim):
                    b.append(0.0)
                    rows.append(row_idx)
                    cols.append(slc.start + k)
                    data.append(scale)
                    row_idx += 1

        if row_idx == 0:
            trans_solution = np.zeros(trans_params, dtype=float)
        else:
            mat = sparse.coo_matrix(
                (data, (rows, cols)), shape=(row_idx, trans_params)
            ).tocsr()
            b_vec = np.asarray(b, dtype=float)
            trans_solution = lsqr(mat, b_vec, **lsqr_kwargs)[0]

        translations = _unpack_solution(
            nodes, trans_slices, trans_solution, ref_node, ndim
        )

        return translations, rotations, scale_vals

    t_pass1, r_pass1, s_pass1 = solve_pass(edges)

    template = g_reg_component_tp.edges[
        next(iter(g_reg_component_tp.edges))
    ]["transform"]
    if isinstance(template, xr.DataArray):
        x_in = template.coords["x_in"]
        x_out = template.coords["x_out"]
    else:
        x_in = None
        x_out = None

    params_pass1 = _build_params_from_components(
        nodes,
        t_pass1,
        r_pass1,
        s_pass1,
        transform,
        ndim,
        x_in=x_in,
        x_out=x_out,
    )

    # Compute residuals in physical units for pruning/reporting.
    residuals_by_edge = compute_edge_residuals(
        g_reg_component_tp, params_pass1, ndim
    )
    metrics, residuals = _compute_edge_metrics(edges, residuals_by_edge)

    residuals = np.asarray(residuals, dtype=float)
    finite_mask = np.isfinite(residuals)
    finite_residuals = residuals[finite_mask]

    if residual_threshold is not None:
        threshold = float(residual_threshold)
    elif finite_residuals.size:
        median = float(np.median(finite_residuals))
        mad = float(np.median(np.abs(finite_residuals - median)))
        threshold = median + float(mad_k) * mad
    else:
        threshold = np.inf

    residuals_for_keep = residuals.copy()
    residuals_for_keep[~np.isfinite(residuals_for_keep)] = np.inf
    keep_mask = (
        residuals_for_keep <= threshold
        if len(residuals_for_keep)
        else np.array([])
    )

    kept_edges = set()
    if keep_mst and len(edges):
        mst_graph = nx.Graph()
        for edge, residual in zip(edges, residuals_for_keep):
            mst_graph.add_edge(edge["u"], edge["v"], weight=residual)
        mst = nx.minimum_spanning_tree(mst_graph, weight="weight")
        kept_edges.update(tuple(sorted(e)) for e in mst.edges)

    final_edges = []
    for idx, (edge, keep) in enumerate(zip(edges, keep_mask)):
        edge_key = tuple(sorted((edge["u"], edge["v"])))
        keep_edge = bool(keep) or edge_key in kept_edges
        metrics[idx]["kept_pass2"] = keep_edge
        if keep_edge:
            final_edges.append(edge)
            kept_edges.add(edge_key)

    if not final_edges:
        final_edges = edges
        kept_edges = {tuple(sorted((e["u"], e["v"]))) for e in edges}
        for metric in metrics:
            metric["kept_pass2"] = True

    t_final, r_final, s_final = solve_pass(final_edges)

    params = _build_params_from_components(
        nodes,
        t_final,
        r_final,
        s_final,
        transform,
        ndim,
        x_in=x_in,
        x_out=x_out,
    )

    metrics_df = pd.DataFrame(metrics) if metrics else None
    info_dict = {
        "metrics": metrics_df,
        "used_edges": list(kept_edges),
    }
    return params, info_dict
