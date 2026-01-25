import logging

import networkx as nx
import pandas as pd
import xarray as xr

from multiview_stitcher import mv_graph, param_utils
from .global_optimization import groupwise_resolution_global_optimization
from .linear_two_pass import groupwise_resolution_linear_two_pass
from .shortest_paths import groupwise_resolution_shortest_paths
from .utils import (
    compute_edge_residuals,
    get_graph_ndim,
    get_graph_timepoints,
    get_reg_graph_with_single_tp_transforms,
)

logger = logging.getLogger(__name__)

_GROUPWISE_RESOLUTION_METHODS = {}


def register_groupwise_resolution_method(name, resolver):
    """
    Register a groupwise resolution method.

    The resolver is called once per connected component and must implement:
    resolver(g_reg_component_tp, **kwargs) -> (params, info_dict).
    The input graph contains single-timepoint transforms.
    """
    if not callable(resolver):
        raise TypeError("Resolver must be callable.")
    _GROUPWISE_RESOLUTION_METHODS[name] = resolver


def _get_groupwise_resolution_method(method):
    if callable(method):
        return method
    if method in _GROUPWISE_RESOLUTION_METHODS:
        return _GROUPWISE_RESOLUTION_METHODS[method]
    raise ValueError(f"Unknown groupwise optimization method: {method}")


def groupwise_resolution(g_reg, method="global_optimization", **kwargs):
    """
    Resolve global parameters by running a method per connected component
    and timepoint.

    Parameters
    ----------
    method : str or Callable
        Name of a registered method, or a callable implementing the
        component-level, single-timepoint resolver API.
    """
    resolver = _get_groupwise_resolution_method(method)
    if not len(g_reg.edges):
        raise (
            mv_graph.NotEnoughOverlapError(
                "Not enough overlap between views for stitching."
            )
        )

    # if only two views, set reference view to the first view
    # this is compatible with a [fixed, moving] convention
    if "reference_view" not in kwargs and len(g_reg.nodes) == 2:
        kwargs["reference_view"] = min(list(g_reg.nodes))

    params = {node: [] for node in g_reg.nodes}
    info_metrics = []
    used_edges_by_t = {}

    # Resolve per timepoint and connected component, then stitch results.
    t_coords = get_graph_timepoints(g_reg)

    # Normalize to a single-timepoint loop for uniform handling.
    iter_t_coords = t_coords if t_coords else [None]
    for it, t in enumerate(iter_t_coords):
        g_reg_t = (
            get_reg_graph_with_single_tp_transforms(g_reg, t)
            if t is not None
            else g_reg
        )
        for icc, cc in enumerate(nx.connected_components(g_reg_t)):
            g_reg_subgraph = g_reg_t.subgraph(list(cc))
            if not g_reg_subgraph.number_of_edges():
                ndim = get_graph_ndim(g_reg_subgraph)
                cc_params = {
                    node: param_utils.identity_transform(ndim)
                    for node in cc
                }
                cc_info = None
            else:
                cc_params, cc_info = resolver(g_reg_subgraph, **kwargs)
            for node in cc:
                params[node].append(cc_params[node])

            if cc_info is not None:
                # Accumulate metrics and edge usage from the resolver.
                metrics = cc_info.get("metrics")
                if metrics is not None:
                    metrics = metrics.copy()
                    if t is not None:
                        metrics["t"] = [t] * len(metrics)
                    if "icc" not in metrics.columns:
                        metrics["icc"] = [icc] * len(metrics)
                    info_metrics.append(metrics)
                used_edges = cc_info.get("used_edges")
                if used_edges is not None:
                    used_edges_by_t.setdefault(it, set()).update(
                        tuple(sorted(e)) for e in used_edges
                    )

    # Concatenate per-timepoint parameters.
    if t_coords:
        params = {
            node: xr.concat(params[node], dim="t").assign_coords(
                {"t": t_coords}
            )
            for node in params
        }
    else:
        params = {node: params[node][0] for node in params}

    edge_residuals_by_t = {}
    for it, t in enumerate(iter_t_coords):
        params_t = {
            node: (
                params[node].sel({"t": t_coords[it]})
                if isinstance(params[node], xr.DataArray)
                and "t" in params[node]
                else params[node]
            )
            for node in params
        }
        g_reg_t = (
            get_reg_graph_with_single_tp_transforms(g_reg, t)
            if t is not None
            else g_reg
        )
        edge_residuals_by_t[it] = compute_edge_residuals(
            g_reg_t, params_t
        )

    info_dict = {
        "metrics": pd.concat(info_metrics) if info_metrics else None,
        "edge_residuals": edge_residuals_by_t,
        "used_edges": {k: list(v) for k, v in used_edges_by_t.items()},
    }
    return params, info_dict


register_groupwise_resolution_method(
    "global_optimization", groupwise_resolution_global_optimization
)
register_groupwise_resolution_method(
    "shortest_paths", groupwise_resolution_shortest_paths
)
register_groupwise_resolution_method(
    "linear_two_pass", groupwise_resolution_linear_two_pass
)

__all__ = [
    "groupwise_resolution",
    "groupwise_resolution_global_optimization",
    "groupwise_resolution_shortest_paths",
    "groupwise_resolution_linear_two_pass",
    "register_groupwise_resolution_method",
]
