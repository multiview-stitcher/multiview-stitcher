# Global parameter resolution

After pairwise registration, global parameter resolution computes a consistent
transform for each view. Select a built-in method by name via
`registration.register(..., groupwise_resolution_method=...)` and pass options
through `groupwise_resolution_kwargs`. Custom methods are described in the
Extension API section.

## `global_optimization` (default)

- Iterative optimization on a virtual-bead graph derived from pairwise overlaps.
- Supports `transform` modes: translation, rigid, similarity, affine.
- Prunes edges based on residuals to reject outliers.
- Robust but can be slower for large graphs.

## `linear_two_pass`

- Fast sparse least-squares solve with a two-pass outlier pruning step.
- Supports `transform` modes: translation, rigid, similarity.
- Key options: `residual_threshold` or `mad_k`, `keep_mst`,
  `weight_mode` (quality/overlap weighting).
- Good for large datasets where speed and robust pruning matter.

## `groupwise_resolution`

- Sparse Laplacian-style solves with first-order rotation/scale linearization
  and two-pass outlier pruning.
- Supports `transform` modes: translation, rigid, similarity.
- Key options: `residual_threshold` or `mad_k`, `keep_mst`, `weight_mode`,
  and `prior_lambda` (stage-frame regularizer).
- Uses physical residuals for pruning; designed for fast global correction.

## `shortest_paths`

- Builds global transforms by concatenating pairwise transforms along
  quality-weighted shortest paths.
- Very fast and deterministic.
- Best when the registration graph is tree-like or pairwise registrations are
  reliable; drift can accumulate in loopy graphs.
