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
