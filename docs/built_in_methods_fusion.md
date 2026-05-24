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

### `multi_view_deconvolution`

Implements the efficient Bayesian multi-view deconvolution method from:

> Preibisch et al., *"Efficient Bayesian-based multiview deconvolution"*,
> Nature Methods 11, 645–648 (2014). <https://doi.org/10.1038/nmeth.2929>

Iteratively refines a single fused estimate *ψ* by applying, for each view in
turn, a forward blur, a ratio correction, and a back-projected multiplicative
update.  Blending weights gate updates so that only views with genuine coverage
contribute at each pixel.

**Key parameters** (passed via `fusion_func_kwargs`):

| Parameter | Default | Description |
|---|---|---|
| `psfs` | `None` | List of per-view PSF arrays (one per view). When `None`, a Gaussian PSF is estimated from `output_spacing`, `na`, and `wavelength_um`. |
| `psf_type` | `"EFFICIENT_BAYESIAN"` | Compound back-projection kernel variant: `"EFFICIENT_BAYESIAN"` (most accurate), `"OPTIMIZATION_I"`, `"OPTIMIZATION_II"`, or `"INDEPENDENT"` (standard RL, no cross-view coupling). |
| `n_iterations` | `10` | Number of deconvolution iterations. |
| `lambda_reg` | `0.0` | Tikhonov regularisation strength; `0` disables it. Values in `1e-4`–`1e-2` give mild smoothing. |
| `output_spacing` | `None` | Physical pixel spacing (µm) keyed by dimension, used for automatic PSF estimation. |
| `na` | `0.8` | Numerical aperture for PSF estimation. |
| `wavelength_um` | `0.5` | Emission wavelength in µm for PSF estimation. |
| `sample_boundary_erosion_px` | `0` | Pixels to erode the union coverage mask before zeroing output; removes the bright-ring artefact at the outer sample boundary. |

**GPU support:** When the input arrays are `cupy` arrays, convolutions run on
the GPU via `cupyx.scipy.ndimage`. Use `use_cupy=True` in `fusion.fuse` to
enable this automatically.

**Chunk overlap:** `multi_view_deconvolution` declares a
`required_overlap` equal to the PSF half-width, ensuring chunked fusion does
not introduce seam artefacts at block boundaries.

**Minimal usage example:**

```python
from multiview_stitcher import fusion
from multiview_stitcher.fusion import multi_view_deconvolution

fused = fusion.fuse(
    images=sims,
    transform_key="affine_registered",
    fusion_func=multi_view_deconvolution,
    fusion_func_kwargs=dict(
        n_iterations=10,
        output_spacing={"z": 0.5, "y": 0.13, "x": 0.13},
        na=0.8,
        wavelength_um=0.52,
    ),
    use_cupy=True,  # optional – omit for CPU-only
)
```

## Blending weights

`fusion.fuse` computes smooth blending weights using
`weights.get_blending_weights`. The falloff is controlled by `blending_widths`
(physical units per dimension) and normalized across views.

## Fusion weights

Additional per-view weights can be supplied via `weights_func` when the fusion
method accepts `fusion_weights`.

- `weights.content_based`: Content-based weights (Preibisch et al.) computed
  from local contrast; configure with `sigma_1` and `sigma_2`.
