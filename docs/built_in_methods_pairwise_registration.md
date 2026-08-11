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
  `itk-elastix` dependency: `pip install multiview-stitcher[itk-elastix]`, or
  `itkwasm-elastix` &mdash; see [Backends](#backends-and-elastix-in-the-browser)).
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
| `log_to_console` | bool | `False` | Print elastix logging output to the console (useful for debugging). Only the `itk` backend takes it. |
| `backend` | str | auto | Which elastix to run: `"itk"` (the `itk-elastix` package) or `"itkwasm"` (the WebAssembly build). By default `itk-elastix` where it is installed, and the WebAssembly build otherwise. |

Any additional keyword arguments are forwarded to the backend's registration
call, i.e. to `itk.elastix_registration_method` for the `itk` backend.

### Backends, and elastix in the browser

`itk-elastix` is a native extension with no WebAssembly build, so it cannot
run in the browser. The same elastix compiled to WebAssembly is published as
[`itkwasm-elastix`](https://pypi.org/project/itkwasm-elastix/), and
`registration_ITKElastix` runs on either.

The two share everything that decides the registration - the stages, the
parameter maps, the initial transform and how the resulting affine is read
back, all of which live in `multiview_stitcher.elastix` - and differ only in
three things: how an image is made, where a transform's defaults come from and
how one stage is run. Those three are what a *backend* provides
(`browser.elastix.ITKWasmElastixBackend` for the WebAssembly one), so a change
to how this package registers with elastix takes effect in both runtimes at
once.

In the browser the method is selectable under Registration &rarr; Advanced
&rarr; Pairwise reg (see
[Stitching in the browser](stitching_in_the_browser.md#registration-methods));
it is `browser.elastix.registration_elastix`, which is this same function with
`backend="itkwasm"` pinned.

The WebAssembly backend works on CPython too, where itk-wasm runs the module
through wasmtime - which is how it is tested:

```python
registration.register(
    msims,
    pairwise_reg_func=registration.registration_ITKElastix,
    pairwise_reg_func_kwargs={
        "backend": "itkwasm",
        "transform_types": ["translation", "rigid"],
    },
)
```
