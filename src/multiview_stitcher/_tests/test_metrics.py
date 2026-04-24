"""
Tests for the metrics module.
"""

import numpy as np
import pytest
from matplotlib import pyplot as plt

from multiview_stitcher import metrics, msi_utils, param_utils, sample_data, vis_utils


# ---------------------------------------------------------------------------
# NCC unit tests
# ---------------------------------------------------------------------------


def test_ncc_identical():
    """NCC of two identical images is 1."""
    rng = np.random.default_rng(0)
    im = rng.random((20, 20))
    assert np.isclose(metrics.normalized_cross_correlation(im, im), 1.0)


def test_ncc_anticorrelated():
    """NCC of an image and its negative is -1 (up to floating point)."""
    rng = np.random.default_rng(0)
    im = rng.random((20, 20))
    assert np.isclose(metrics.normalized_cross_correlation(im, -im), -1.0)


def test_ncc_constant_image():
    """NCC with a constant image returns NaN (undefined)."""
    im = np.ones((20, 20))
    rng = np.random.default_rng(0)
    other = rng.random((20, 20))
    result = metrics.normalized_cross_correlation(im, other)
    assert np.isnan(result)


def test_ncc_all_nan():
    """NCC when all pixels are NaN returns NaN."""
    im = np.full((10, 10), np.nan)
    result = metrics.normalized_cross_correlation(im, im)
    assert np.isnan(result)


def test_ncc_partial_nan():
    """NCC ignores NaN pixels and still returns a valid value."""
    rng = np.random.default_rng(0)
    im = rng.random((20, 20))
    im_nan = im.copy()
    im_nan[:5, :] = np.nan  # mask top rows in both images identically
    im_nan2 = im_nan.copy()
    result = metrics.normalized_cross_correlation(im_nan, im_nan2)
    assert not np.isnan(result)
    assert np.isclose(result, 1.0)


# ---------------------------------------------------------------------------
# Main correctness test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ndim", [2, 3])
def test_tile_pair_image_metrics_aligned_beats_misaligned(ndim):
    """
    A correctly aligned transform key yields higher NCC than misaligned ones,
    both per-pair and in the summary statistics.

    Setup
    -----
    * Two adjacent tiles generated from the same ground-truth image
      (no noise, no drift, no intra-tile shift).  The overlap region
      therefore contains exactly the same content → NCC ≈ 1 for the
      correct alignment.
    * Two extra transform keys are added:
        - ``misaligned_t``: a pure translation perturbation that differs
          between tiles, breaking relative alignment.
        - ``misaligned_r``: a rigid (rotation + translation) perturbation.
    """
    base_transform_key = "ground_truth"

    # -----------------------------------------------------------------------
    # Build a simple tiled dataset: two tiles sharing an overlap
    # -----------------------------------------------------------------------
    if ndim == 2:
        sims = sample_data.generate_tiled_dataset(
            ndim=ndim,
            N_c=1,
            N_t=1,
            tile_size=60,
            tiles_x=2,
            tiles_y=2,
            overlap=20,
            zoom=6,
            shift_scale=0.0,
            drift_scale=0.0,
            transform_key=base_transform_key,
        )
        shift_amount = 4.0  # physical units

    else:  # ndim == 3
        sims = sample_data.generate_tiled_dataset(
            ndim=ndim,
            N_c=1,
            N_t=1,
            tile_size=30,
            tiles_x=2,
            tiles_y=2,
            tiles_z=1,
            overlap=10,
            zoom=3,
            shift_scale=0.0,
            drift_scale=0.0,
            transform_key=base_transform_key,
        )
        shift_amount = 2.0

    msims = [msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims]

    # -----------------------------------------------------------------------
    # Register two misaligned transform keys using random_affine.
    # Each tile receives an independent random transform so that the
    # relative misalignment between adjacent tiles is non-trivial.
    # -----------------------------------------------------------------------
    np.random.seed(0)
    misaligned_t_affines = [
        param_utils.random_affine(
            ndim=ndim,
            translation_scale=shift_amount,
            rotation_scale=0.0,
            scale_scale=0.0,
        )
        for _ in msims
    ]
    misaligned_r_affines = [
        param_utils.random_affine(
            ndim=ndim,
            translation_scale=shift_amount,
            rotation_scale=0.12,
            scale_scale=0.0,
        )
        for _ in msims
    ]

    for i, msim in enumerate(msims):
        msi_utils.set_affine_transform(
            msim,
            param_utils.affine_to_xaffine(misaligned_t_affines[i]),
            transform_key="misaligned_t",
            base_transform_key=None,
        )
        msi_utils.set_affine_transform(
            msim,
            param_utils.affine_to_xaffine(misaligned_r_affines[i]),
            transform_key="misaligned_r",
            base_transform_key=None,
        )

    # -----------------------------------------------------------------------
    # Compute metrics
    # -----------------------------------------------------------------------
    result = metrics.tile_pair_image_metrics(
        msims,
        base_transform_key=base_transform_key,
        query_transform_keys=[base_transform_key, "misaligned_t", "misaligned_r"],
        metric_funcs={"ncc": metrics.normalized_cross_correlation},
        max_tolerance=None,
    )

    # -----------------------------------------------------------------------
    # Assertions: correct alignment must have strictly higher NCC
    # -----------------------------------------------------------------------
    pairs_dict = result["pairs"]
    assert len(pairs_dict) > 0, "No pairs were computed"

    for (fixed_idx, moving_idx), pair_metrics in pairs_dict.items():
        ncc_base = pair_metrics[base_transform_key]["ncc"]
        ncc_t = pair_metrics["misaligned_t"]["ncc"]
        ncc_r = pair_metrics["misaligned_r"]["ncc"]

        # Skip pairs where the metric is NaN (degenerate overlap)
        if np.isnan(ncc_base):
            continue

        assert ncc_base > ncc_t, (
            f"Pair ({fixed_idx} → {moving_idx}): "
            f"base NCC {ncc_base:.4f} should exceed "
            f"misaligned_t NCC {ncc_t:.4f}"
        )
        assert ncc_base > ncc_r, (
            f"Pair ({fixed_idx} → {moving_idx}): "
            f"base NCC {ncc_base:.4f} should exceed "
            f"misaligned_r NCC {ncc_r:.4f}"
        )

    summary = result["summary"]
    ncc_base_summary = summary[base_transform_key]["ncc"]
    ncc_t_summary = summary["misaligned_t"]["ncc"]
    ncc_r_summary = summary["misaligned_r"]["ncc"]

    assert ncc_base_summary > ncc_t_summary, (
        f"Summary: base NCC {ncc_base_summary:.4f} should exceed "
        f"misaligned_t NCC {ncc_t_summary:.4f}"
    )
    assert ncc_base_summary > ncc_r_summary, (
        f"Summary: base NCC {ncc_base_summary:.4f} should exceed "
        f"misaligned_r NCC {ncc_r_summary:.4f}"
    )


# ---------------------------------------------------------------------------
# Structural / API tests
# ---------------------------------------------------------------------------


def test_tile_pair_image_metrics_return_structure():
    """tile_pair_image_metrics returns a dict with 'pairs' and 'summary' keys."""
    base_transform_key = "gt"
    sims = sample_data.generate_tiled_dataset(
        ndim=2, N_c=1, N_t=1, tile_size=40, tiles_x=2, tiles_y=1,
        overlap=10, zoom=4, shift_scale=0.0, drift_scale=0.0,
        transform_key=base_transform_key,
    )
    msims = [msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims]

    result = metrics.tile_pair_image_metrics(
        msims,
        base_transform_key=base_transform_key,
        query_transform_keys=base_transform_key,  # single str, not list
        metric_funcs={"ncc": metrics.normalized_cross_correlation},
    )

    assert "pairs" in result
    assert "summary" in result

    # Two tiles, bidirectional=False (default) → one directed edge: (0→1)
    assert len(result["pairs"]) == 1

    for pair, pair_metrics in result["pairs"].items():
        assert isinstance(pair, tuple) and len(pair) == 2
        assert base_transform_key in pair_metrics
        assert "ncc" in pair_metrics[base_transform_key]
        val = pair_metrics[base_transform_key]["ncc"]
        assert isinstance(val, (float, np.floating))

    assert base_transform_key in result["summary"]
    assert "ncc" in result["summary"][base_transform_key]


def test_tile_pair_image_metrics_custom_metric_func():
    """Custom metric functions are applied correctly."""
    base_transform_key = "gt"
    sims = sample_data.generate_tiled_dataset(
        ndim=2, N_c=1, N_t=1, tile_size=40, tiles_x=2, tiles_y=1,
        overlap=10, zoom=4, shift_scale=0.0, drift_scale=0.0,
        transform_key=base_transform_key,
    )
    msims = [msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims]

    def mean_abs_diff(im1, im2):
        mask = ~(np.isnan(im1) | np.isnan(im2))
        if not np.any(mask):
            return np.nan
        return float(np.mean(np.abs(im1[mask] - im2[mask])))

    result = metrics.tile_pair_image_metrics(
        msims,
        base_transform_key=base_transform_key,
        query_transform_keys=[base_transform_key],
        metric_funcs={"ncc": metrics.normalized_cross_correlation, "mad": mean_abs_diff},
    )

    for pair_metrics in result["pairs"].values():
        assert "ncc" in pair_metrics[base_transform_key]
        assert "mad" in pair_metrics[base_transform_key]
    assert "ncc" in result["summary"][base_transform_key]
    assert "mad" in result["summary"][base_transform_key]


def test_tile_pair_image_metrics_max_tolerance():
    """Applying max_tolerance shrinks the comparison bbox without error."""
    base_transform_key = "gt"
    sims = sample_data.generate_tiled_dataset(
        ndim=2, N_c=1, N_t=1, tile_size=60, tiles_x=2, tiles_y=1,
        overlap=20, zoom=6, shift_scale=0.0, drift_scale=0.0,
        transform_key=base_transform_key,
    )
    msims = [msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims]

    # Small tolerance: bbox should still be valid
    result = metrics.tile_pair_image_metrics(
        msims,
        base_transform_key=base_transform_key,
        query_transform_keys=[base_transform_key],
        max_tolerance=1.0,
    )
    assert "pairs" in result
    assert len(result["pairs"]) == 1


def test_tile_pair_image_metrics_with_spacing():
    """Custom spacing parameter is accepted and produces valid results."""
    base_transform_key = "gt"
    sims = sample_data.generate_tiled_dataset(
        ndim=2, N_c=1, N_t=1, tile_size=60, tiles_x=2, tiles_y=1,
        overlap=20, zoom=6, shift_scale=0.0, drift_scale=0.0,
        transform_key=base_transform_key,
    )
    msims = [msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims]

    result = metrics.tile_pair_image_metrics(
        msims,
        base_transform_key=base_transform_key,
        query_transform_keys=[base_transform_key],
        spacing=1.0,  # coarser spacing → faster computation
    )
    assert "pairs" in result


# ---------------------------------------------------------------------------
# Visualisation tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ndim", [2, 3])
def test_plot_tile_pair_image_metrics(ndim, monkeypatch):
    """plot_tile_pair_image_metrics returns one (fig, ax) per query key without error."""
    monkeypatch.setattr(plt, "show", lambda: None)

    base_transform_key = "gt"
    sims = sample_data.generate_tiled_dataset(
        ndim=ndim,
        N_c=1,
        N_t=1,
        tile_size=30 if ndim == 3 else 40,
        tiles_x=2,
        tiles_y=2,
        tiles_z=1 if ndim == 3 else None,
        overlap=10,
        zoom=3 if ndim == 3 else 4,
        shift_scale=0.0,
        drift_scale=0.0,
        transform_key=base_transform_key,
    )
    msims = [msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims]

    query_keys = [base_transform_key]
    result = metrics.tile_pair_image_metrics(
        msims,
        base_transform_key=base_transform_key,
        query_transform_keys=query_keys,
    )

    for show_bboxes in [True, False]:
        plots = vis_utils.plot_tile_pair_image_metrics(
            msims,
            result,
            base_transform_key=base_transform_key,
            query_transform_keys=query_keys,
            show_bboxes=show_bboxes,
        )

        assert set(plots.keys()) == set(query_keys)
        for q, (fig, ax) in plots.items():
            assert fig is not None
            assert ax is not None

        plt.close("all")
