# Fusion

Fusion combines the pre-transformed views into a single output image. Built-in
methods are selected with `fusion.fuse(..., fusion_func=..., weights_func=...)`.
Custom fusion and weight functions are described in the Extension API section.

## Fusion methods

### `weighted_average_fusion` (default)

- Uses blending weights with optional extra `fusion_weights`.
- Good general-purpose choice for smooth transitions across overlaps.

### `simple_average_fusion`

- Averages valid pixels (ignores NaNs), without blending weights.
- Useful as a baseline when weights are not desired.

### `max_fusion`

- Pixel-wise maximum across views.
- Useful for sparse or bright features where max intensity is preferred.

## Blending weights

`fusion.fuse` computes smooth blending weights using
`weights.get_blending_weights`. The falloff is controlled by `blending_widths`
(physical units per dimension) and normalized across views.

## Fusion weights

Additional per-view weights can be supplied via `weights_func` when the fusion
method accepts `fusion_weights`.

- `weights.content_based`: Content-based weights (Preibisch et al.) computed
  from local contrast; configure with `sigma_1` and `sigma_2`.
