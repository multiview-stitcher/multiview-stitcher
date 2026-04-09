# Pairwise registration

Multiview-stitcher ships with built-in pairwise registration functions that can
be selected in `registration.register` via `pairwise_reg_func`. For custom
implementations, see the Extension API section.

## `phase_correlation_registration` (default)

- Pixel-space, translation-only registration based on
  `skimage.registration.phase_cross_correlation`.
- Handles NaNs in the overlap; automatically chooses union/intersection
  disambiguation based on valid pixels.
- Sub-pixel refinement is enabled via `upsample_factor` (defaults to 10 in 2D,
  2 in 3D).
- Fast and robust for translational overlaps.

## `registration_ANTsPy`

- Physical-space registration using ANTsPy (requires the optional `antspyx`
  dependency).
- Runs a sequence of transform stages (default: Translation, Rigid,
  Similarity) starting from the passed `initial_affine`.
- Respects image spacing and origin; useful when simple translation is not
  sufficient.
- Configure via `pairwise_reg_func_kwargs`, e.g. `transform_types`,
  `aff_metric`, or `aff_iterations`.

## `registration_ITKElastix`

- Physical-space registration using ITKElastix (requires the optional
  `itk-elastix` dependency: `pip install multiview-stitcher[itk-elastix]`).
- Runs a sequence of transform stages (default: Translation, Rigid) starting from the passed `initial_affine`. Supported stages: `'Translation'`, `'Rigid'`, `'Similarity'`, `'Affine'` (case-insensitive).
- Respects image spacing and origin; each stage threads its result forward as the initial transform for the next stage.
- Configure via `pairwise_reg_func_kwargs`, for example:

```python
registration.register(
    msims,
    pairwise_reg_func=registration.registration_ITKElastix,
    pairwise_reg_func_kwargs={
        "transform_types": ["Rigid", "Affine"],
        "number_of_resolutions": 3,
        "number_of_iterations": 500,
        "metric": "AdvancedMattesMutualInformation",
    },
    groupwise_resolution_kwargs={
        "transform": "Affine",  # this typically equals the last transform type in pairwise_reg_func_kwargs
    },
)
```

### Options

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `transform_types` | list of str | `["Translation", "Rigid"]` | Sequence of transform stages to run. Each stage feeds its result as the initial transform for the next. Supported values: `"Translation"`, `"Rigid"`, `"Similarity"`, `"Affine"` (case-insensitive). |
| `number_of_resolutions` | int | `2` | Number of resolution levels in the multi-resolution pyramid. Higher values let the optimizer start from coarser scales, improving convergence on large shifts. |
| `number_of_iterations` | int | elastix default | Maximum optimizer iterations per resolution level. If not set, the elastix default for the chosen transform type is used. |
| `metric` | str | elastix default | Similarity metric. Common choices: `"AdvancedMattesMutualInformation"` (good for multi-modal), `"AdvancedNormalizedCorrelation"`, `"AdvancedMeanSquares"`, `"NormalizedMutualInformation"`. |
| `log_to_console` | bool | `False` | Print elastix logging output to the console (useful for debugging). |

Any additional keyword arguments are forwarded directly to
`itk.elastix_registration_method`.
