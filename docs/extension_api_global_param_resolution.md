# Global parameter resolution

Custom functions for global parameter resolution can be registered via
`multiview_stitcher.registration.register_groupwise_resolution_method` and
selected by name in `registration.register` via `groupwise_resolution_method`.

The resolver is called once per connected component and timepoint. The
input is a NetworkX `Graph` containing only a single connected component
and single-timepoint transforms.

## Registration graph

The input registration graph `g_reg_component_tp` is a `networkx.Graph` with:

- **Nodes**: view/tile indices. Each node has `stack_props` describing the view,
  including `spacing` (a dict of spatial dimension keys and spacing values).
- **Edges**: pairwise registration links with attributes:
  - `transform`: `xarray.DataArray` affine transform (dims `x_in`, `x_out`), mapping
    fixed to moving view coordinates. The graph passed to the resolver has
    a single timepoint, so there is no `t` dimension.
  - `quality`: registration quality score (float or 1D array).
  - `overlap`: overlap area/volume.
  - `bbox`: overlap bounding box used for registration.

## Resolver API

```python
from multiview_stitcher import registration

def custom_groupwise_resolution(
    g_reg_component_tp,  # networkx.Graph, single CC, single timepoint
    reference_view, # a reference node is passed (either specified by the user or determined beforehand)
    **kwargs,
) -> tuple[dict, dict]:

    # Compute per-node transforms.
    params = {
        node: xparams,  # xr.DataArray, dims x_in/x_out
        for node in g_reg_component_tp.nodes
    }

    # Optional metadata for summary plots/metrics.
    info = {
        "metrics": metrics_df,  # pd.DataFrame or None
        "used_edges": used_edges,  # list of (u, v) edges to keep
        "edge_residuals": edge_residuals,  # dict[(u, v)] -> float
    }
    return params, info

registration.register_groupwise_resolution_method(
    "my_method", custom_groupwise_resolution
)
```

`registration.groupwise_resolution` will collect results across all timepoints
and connected components. If the input data contains timepoints, the returned
params for each node are concatenated along `t` in the output of
`registration.register`.

`reference_view` is passed to the resolver (via `groupwise_resolution_kwargs`
in `registration.register`) and can be used to fix one node as a reference in
the component.

The framework creates `optimized_graph_t0` (the first-timepoint graph) by
copying the input registration graph, pruning any edges not listed in
`used_edges`, and storing per-node transforms and `edge_residuals` as node/edge
attributes.
