import numpy as np
import pandas as pd
import xarray as xr
import networkx as nx
from scipy import sparse
from scipy.sparse.linalg import lsqr
from scipy.spatial.transform import Rotation

from multiview_stitcher import mv_graph, param_utils


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


def _cross_matrix_for_w_cross_d(dvec, ndim):
    """Matrix C(d) such that C(d) @ w = w x d (2D/3D)."""
    if ndim == 2:
        return np.array([[-dvec[1]], [dvec[0]]], dtype=float)
    return np.array(
        [
            [0.0, dvec[2], -dvec[1]],
            [-dvec[2], 0.0, dvec[0]],
            [dvec[1], -dvec[0], 0.0],
        ],
        dtype=float,
    )


def _build_node_slices(nodes, reference_view, ndim, rot_dim, use_rot, use_scale):
    """
    Assign parameter vector slices for each node (excluding reference).

    Ordering per node: translation, rotation (optional), scale (optional).
    """
    index = 0
    slices = {}
    for node in nodes:
        if node == reference_view:
            continue
        slices[node] = {}
        slices[node]["t"] = slice(index, index + ndim)
        index += ndim
        if use_rot:
            slices[node]["rot"] = slice(index, index + rot_dim)
            index += rot_dim
        if use_scale:
            slices[node]["scale"] = slice(index, index + 1)
            index += 1
    return slices, index


def _solve_linear_system(
    edges,
    node_slices,
    n_params,
    reference_view,
    ndim,
    rot_dim,
    use_rot,
    use_scale,
    **lsqr_kwargs,
):
    """
    Build and solve the sparse linear system for all edge constraints.

    Returns the packed solution vector for all nodes (reference fixed to 0).
    """
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
        dvec = edge["trans"]

        if use_rot:
            rot_uv = edge["rot"]
            for k in range(rot_dim):
                b.append(scale * rot_uv[k])
                if u != reference_view:
                    rows.append(row_idx)
                    cols.append(node_slices[u]["rot"].start + k)
                    data.append(scale)
                if v != reference_view:
                    rows.append(row_idx)
                    cols.append(node_slices[v]["rot"].start + k)
                    data.append(-scale)
                row_idx += 1

        if use_scale:
            b.append(scale * edge["scale"])
            if u != reference_view:
                rows.append(row_idx)
                cols.append(node_slices[u]["scale"].start)
                data.append(scale)
            if v != reference_view:
                rows.append(row_idx)
                cols.append(node_slices[v]["scale"].start)
                data.append(-scale)
            row_idx += 1

        for k in range(ndim):
            b.append(scale * dvec[k])
            if u != reference_view:
                rows.append(row_idx)
                cols.append(node_slices[u]["t"].start + k)
                data.append(scale)
            if v != reference_view:
                rows.append(row_idx)
                cols.append(node_slices[v]["t"].start + k)
                data.append(-scale)
            if use_rot and v != reference_view:
                cross = edge["cross"]
                for j in range(rot_dim):
                    coeff = -cross[k, j] * scale
                    if coeff != 0.0:
                        rows.append(row_idx)
                        cols.append(node_slices[v]["rot"].start + j)
                        data.append(coeff)
            if use_scale and v != reference_view:
                rows.append(row_idx)
                cols.append(node_slices[v]["scale"].start)
                data.append(-dvec[k] * scale)
            row_idx += 1

    if row_idx == 0:
        return np.zeros(n_params, dtype=float)

    mat = sparse.coo_matrix(
        (data, (rows, cols)), shape=(row_idx, n_params)
    ).tocsr()
    b = np.asarray(b, dtype=float)
    result = lsqr(mat, b, **lsqr_kwargs)
    return result[0]


def _vector_to_node_params(
    nodes,
    node_slices,
    solution,
    reference_view,
    ndim,
    rot_dim,
    use_rot,
    use_scale,
):
    """Unpack the solution vector into per-node translation/rotation/scale."""
    translations = {node: np.zeros(ndim, dtype=float) for node in nodes}
    rotations = (
        {node: np.zeros(rot_dim, dtype=float) for node in nodes}
        if use_rot
        else {node: None for node in nodes}
    )
    scales = (
        {node: 0.0 for node in nodes}
        if use_scale
        else {node: None for node in nodes}
    )

    for node in nodes:
        if node == reference_view:
            continue
        slices = node_slices[node]
        translations[node] = solution[slices["t"]]
        if use_rot:
            rotations[node] = solution[slices["rot"]]
        if use_scale:
            scales[node] = float(solution[slices["scale"]])

    return translations, rotations, scales


def _compute_edge_metrics(
    edges,
    translations,
    rotations,
    scales,
    use_rot,
    use_scale,
    rot_weight,
    scale_weight,
):
    """
    Compute per-edge residuals for pruning and reporting.

    Residuals are computed from linearized constraints using pass-1 params.
    """
    metrics = []
    residuals = []
    for edge in edges:
        u = edge["u"]
        v = edge["v"]
        dvec = edge["trans"]
        cross = edge["cross"]

        t_u = translations[u]
        t_v = translations[v]
        rot_v = rotations[v] if use_rot else np.zeros_like(dvec[:0])
        scale_v = scales[v] if use_scale else 0.0

        cross_term = cross @ rot_v if use_rot else 0.0
        res_t = (t_u - t_v) - cross_term - scale_v * dvec - dvec
        r_trans = float(np.linalg.norm(res_t))

        r_rot = 0.0
        if use_rot:
            rot_u = rotations[u]
            r_rot = float(
                np.linalg.norm((rot_u - rot_v) - edge["rot"])
            )

        r_scale = 0.0
        if use_scale:
            scale_u = scales[u]
            r_scale = float((scale_u - scale_v) - edge["scale"])
            r_scale = abs(r_scale)

        r_total = float(
            np.sqrt(
                r_trans**2
                + (rot_weight * r_rot) ** 2
                + (scale_weight * r_scale) ** 2
            )
        )
        residuals.append(r_total)
        metrics.append(
            {
                "u": u,
                "v": v,
                "weight": edge["weight"],
                "r_trans": r_trans,
                "r_rot": r_rot,
                "r_scale": r_scale,
                "r_total": r_total,
            }
        )
    return metrics, np.asarray(residuals, dtype=float)


def linear_two_pass_groupwise_resolution(
    g_reg_component_tp,
    reference_view=None,
    mode="rigid",
    prune_quantile=0.9,
    keep_mst=True,
    weight_mode="quality_overlap",
    rot_weight=1.0,
    scale_weight=1.0,
    **kwargs,
):
    """
    Fast groupwise resolution using a sparse linear solve with two-pass
    outlier pruning and first-order coupling between translation and
    rotation/scale.
    """
    if not g_reg_component_tp.number_of_edges():
        ndim = _get_graph_ndim(g_reg_component_tp)
        params = {
            node: param_utils.identity_transform(ndim)
            for node in g_reg_component_tp.nodes
        }
        return params, {"metrics": None, "used_edges": []}

    if mode not in ("translation", "rigid", "similarity"):
        raise ValueError(f"Unknown mode: {mode}")

    ndim = _get_graph_ndim(g_reg_component_tp)
    if ndim not in (2, 3):
        raise ValueError("Only 2D and 3D supported.")

    use_rot = mode in ("rigid", "similarity")
    use_scale = mode == "similarity"
    rot_dim = 1 if ndim == 2 else 3

    if reference_view is not None and reference_view in g_reg_component_tp:
        ref_node = reference_view
    else:
        ref_node = mv_graph.get_node_with_maximal_edge_weight_sum_from_graph(
            g_reg_component_tp, weight_key="quality"
        )

    edges = []
    for edge in g_reg_component_tp.edges:
        affine = g_reg_component_tp.edges[edge]["transform"]
        if isinstance(affine, xr.DataArray):
            affine = affine.data
        affine = np.asarray(affine, dtype=float)

        linear = affine[:ndim, :ndim]
        dvec = affine[:ndim, ndim]

        rot_uv = None
        scale_uv = None
        if use_rot or use_scale:
            rmat, scale = _closest_rotation_and_scale(linear, ndim)
            if use_rot:
                rot_uv = _rotation_to_vector(rmat, ndim)
            if use_scale:
                scale_uv = float(np.log(scale))

        edges.append(
            {
                "u": edge[0],
                "v": edge[1],
                "trans": dvec,
                "rot": rot_uv,
                "scale": scale_uv,
                "cross": _cross_matrix_for_w_cross_d(dvec, ndim)
                if use_rot
                else None,
                "weight": _get_edge_weight(
                    g_reg_component_tp.edges[edge], weight_mode
                ),
            }
        )

    nodes = list(g_reg_component_tp.nodes)
    node_slices, n_params = _build_node_slices(
        nodes, ref_node, ndim, rot_dim, use_rot, use_scale
    )

    # Limit solver kwargs to those supported by scipy.sparse.linalg.lsqr.
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

    solution_pass1 = _solve_linear_system(
        edges,
        node_slices,
        n_params,
        ref_node,
        ndim,
        rot_dim,
        use_rot,
        use_scale,
        **lsqr_kwargs,
    )
    t_pass1, r_pass1, s_pass1 = _vector_to_node_params(
        nodes,
        node_slices,
        solution_pass1,
        ref_node,
        ndim,
        rot_dim,
        use_rot,
        use_scale,
    )

    metrics, residuals = _compute_edge_metrics(
        edges,
        t_pass1,
        r_pass1,
        s_pass1,
        use_rot,
        use_scale,
        rot_weight,
        scale_weight,
    )

    if len(residuals):
        threshold = float(np.quantile(residuals, prune_quantile))
    else:
        threshold = 0.0

    keep_mask = residuals <= threshold if len(residuals) else np.array([])

    kept_edges = set()
    if keep_mst and len(edges):
        mst_graph = nx.Graph()
        for edge, residual in zip(edges, residuals):
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

    # Fallback if pruning removed everything.
    if not final_edges:
        final_edges = edges
        kept_edges = {tuple(sorted((e["u"], e["v"]))) for e in edges}
        for metric in metrics:
            metric["kept_pass2"] = True

    solution_pass2 = _solve_linear_system(
        final_edges,
        node_slices,
        n_params,
        ref_node,
        ndim,
        rot_dim,
        use_rot,
        use_scale,
        **lsqr_kwargs,
    )
    t_final, r_final, s_final = _vector_to_node_params(
        nodes,
        node_slices,
        solution_pass2,
        ref_node,
        ndim,
        rot_dim,
        use_rot,
        use_scale,
    )

    template = g_reg_component_tp.edges[
        next(iter(g_reg_component_tp.edges))
    ]["transform"]
    if isinstance(template, xr.DataArray):
        x_in = template.coords["x_in"]
        x_out = template.coords["x_out"]
    else:
        x_in = None
        x_out = None

    params = {}
    for node in nodes:
        if mode == "translation":
            linear = np.eye(ndim)
        elif mode == "rigid":
            if ndim == 2:
                theta = r_final[node][0]
                c = np.cos(theta)
                s = np.sin(theta)
                linear = np.array([[c, -s], [s, c]], dtype=float)
            else:
                linear = Rotation.from_rotvec(r_final[node]).as_matrix()
        else:
            if ndim == 2:
                theta = r_final[node][0]
                c = np.cos(theta)
                s = np.sin(theta)
                rmat = np.array([[c, -s], [s, c]], dtype=float)
            else:
                rmat = Rotation.from_rotvec(r_final[node]).as_matrix()
            linear = np.exp(s_final[node]) * rmat

        matrix = np.eye(ndim + 1, dtype=float)
        matrix[:ndim, :ndim] = linear
        matrix[:ndim, ndim] = t_final[node]
        xparams = param_utils.affine_to_xaffine(matrix)
        if x_in is not None and x_out is not None:
            xparams = xparams.sel(x_in=x_in, x_out=x_out)
        params[node] = xparams

    metrics_df = pd.DataFrame(metrics) if metrics else None
    info_dict = {
        "metrics": metrics_df,
        "used_edges": list(kept_edges),
    }
    return params, info_dict


def _get_graph_ndim(g_reg):
    if g_reg.number_of_edges():
        return (
            g_reg.get_edge_data(*list(g_reg.edges())[0])["transform"].shape[-1]
            - 1
        )
    if len(g_reg.nodes):
        node = next(iter(g_reg.nodes))
        stack_props = g_reg.nodes[node].get("stack_props", {})
        if "spacing" in stack_props:
            return len(stack_props["spacing"].values())
    raise ValueError("Cannot determine dimensionality from graph.")
