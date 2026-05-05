import numpy as np
import dask.array as da
import pytest

from multiview_stitcher import fusion, sample_data
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher.fusion.mv_deconv import (
    PSFType,
    multi_view_deconvolution,
    estimate_psf,
    make_gaussian_psf,
)


# ---------------------------------------------------------------------------
# PSF helpers
# ---------------------------------------------------------------------------

def test_make_gaussian_psf_isotropic():
    psf = make_gaussian_psf(2.0, ndim=2)
    assert psf.ndim == 2
    assert psf.dtype == np.float32
    np.testing.assert_allclose(psf.sum(), 1.0, atol=1e-5)


def test_make_gaussian_psf_anisotropic_3d():
    psf = make_gaussian_psf([3.0, 1.5, 1.5])
    assert psf.ndim == 3
    np.testing.assert_allclose(psf.sum(), 1.0, atol=1e-5)


def test_make_gaussian_psf_custom_shape():
    psf = make_gaussian_psf(2.0, ndim=2, shape=(11, 11))
    assert psf.shape == (11, 11)
    np.testing.assert_allclose(psf.sum(), 1.0, atol=1e-5)


def test_estimate_psf_2d():
    psf = estimate_psf({"y": 0.5, "x": 0.5}, na=0.8, wavelength_um=0.5)
    assert psf.ndim == 2
    np.testing.assert_allclose(psf.sum(), 1.0, atol=1e-5)


def test_estimate_psf_3d():
    psf = estimate_psf({"z": 2.0, "y": 0.5, "x": 0.5}, na=0.8, wavelength_um=0.5)
    assert psf.ndim == 3
    np.testing.assert_allclose(psf.sum(), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# multi_view_deconvolution: direct array interface
# ---------------------------------------------------------------------------

def _make_views(n_views, shape, seed=0):
    rng = np.random.default_rng(seed)
    gt = rng.poisson(100, shape).astype(np.float32)
    views = np.stack(
        [np.clip(gt + rng.normal(0, 5, shape), 0, None) for _ in range(n_views)]
    ).astype(np.float32)
    weights = np.ones((n_views, *shape), dtype=np.float32) / n_views
    return views, weights


@pytest.mark.parametrize("psf_type", list(PSFType))
def test_multi_view_deconvolution_2d_psf_types(psf_type):
    views, weights = _make_views(3, (32, 32))
    result = multi_view_deconvolution(views, weights, psf_type=psf_type, n_iterations=3)
    assert result.shape == (32, 32)
    assert result.dtype == views.dtype
    assert np.all(np.isfinite(result))


def test_multi_view_deconvolution_3d():
    views, weights = _make_views(2, (8, 24, 24))
    result = multi_view_deconvolution(views, weights, n_iterations=2)
    assert result.shape == (8, 24, 24)
    assert np.all(np.isfinite(result))


def test_multi_view_deconvolution_explicit_psfs():
    views, weights = _make_views(2, (32, 32))
    psfs = [make_gaussian_psf(1.0, ndim=2), make_gaussian_psf(2.0, ndim=2)]
    result = multi_view_deconvolution(views, weights, psfs=psfs, n_iterations=3)
    assert result.shape == (32, 32)
    assert np.all(np.isfinite(result))


def test_multi_view_deconvolution_wrong_psf_count_raises():
    views, weights = _make_views(3, (16, 16))
    psfs = [make_gaussian_psf(1.5, ndim=2)]  # wrong: only 1 PSF for 3 views
    with pytest.raises(ValueError, match="n_views"):
        multi_view_deconvolution(views, weights, psfs=psfs, n_iterations=1)


def test_multi_view_deconvolution_tikhonov():
    views, weights = _make_views(2, (24, 24))
    result = multi_view_deconvolution(views, weights, n_iterations=3, lambda_reg=1e-3)
    assert result.shape == (24, 24)
    assert np.all(np.isfinite(result))


def test_multi_view_deconvolution_with_nan_views():
    """Views with NaN padding (outside-FOV) must not pollute the result."""
    views, w = _make_views(2, (32, 32))
    views[0, :, 16:] = np.nan  # half of view 0 outside FOV
    w[0, :, 16:] = 0.0          # zero weight matches
    w[1, :, 16:] = 1.0
    result = multi_view_deconvolution(views, w, n_iterations=3)
    assert np.all(np.isfinite(result))


def test_multi_view_deconvolution_output_spacing():
    views, weights = _make_views(2, (24, 24))
    result = multi_view_deconvolution(
        views, weights,
        output_spacing={"y": 0.5, "x": 0.5},
        na=0.8,
        wavelength_um=0.5,
        n_iterations=2,
    )
    assert result.shape == (24, 24)


# ---------------------------------------------------------------------------
# Integration with fusion.fuse()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ndim", [2, 3])
def test_fuse_pipeline_mv_deconvolution(ndim):
    """multi_view_deconvolution integrates correctly with fusion.fuse()."""
    kwargs = dict(
        ndim=ndim, N_t=1, N_c=1, tile_size=20,
        tiles_x=2, tiles_y=2, overlap=4,
    )
    if ndim == 3:
        kwargs["tiles_z"] = 1

    sims = sample_data.generate_tiled_dataset(**kwargs)

    fused = fusion.fuse(
        sims,
        transform_key="affine_metadata",
        fusion_func=multi_view_deconvolution,
        fusion_method_kwargs=dict(n_iterations=3, psf_sigma_px=1.5),
    )
    result = fused.compute(scheduler="single-threaded")
    assert result.dtype == sims[0].dtype
    assert np.all(np.isfinite(result.data))


@pytest.mark.xfail(
    reason=(
        "Known regression: multi_view_deconvolution creates border-localized "
        "seams at input-view boundaries."
    ),
    strict=True,
)
def test_fuse_pipeline_mv_deconvolution_has_no_tile_border_artefacts():
    """Joint deconvolution should not introduce seam-specific border jumps."""
    np.random.seed(0)
    da.random.seed(0)

    tile_size = 40
    overlap = 30
    sims = sample_data.generate_tiled_dataset(
        ndim=2,
        N_c=1,
        N_t=1,
        overlap=overlap,
        tile_size=tile_size,
        tiles_x=2,
        tiles_y=1,
        tiles_z=1,
        shift_scale=0,
        drift_scale=0,
        zoom=10,
    )

    independently_deconvolved = []
    for sim in sims:
        deconvolved = fusion.fuse(
            [sim],
            fusion_func=multi_view_deconvolution,
            transform_key="affine_metadata",
        )
        independently_deconvolved.append(
            si_utils.get_sim_from_array(
                deconvolved.data,
                dims=deconvolved.dims,
                scale=si_utils.get_spacing_from_sim(sim),
                translation=si_utils.get_origin_from_sim(sim),
            )
        )

    fused_independent = fusion.fuse(
        independently_deconvolved,
        transform_key="affine_metadata",
    )
    fused_joint = fusion.fuse(
        sims,
        fusion_func=multi_view_deconvolution,
        transform_key="affine_metadata",
    )

    profiles = np.array(
        [
            fused_independent.data[0, 0].mean(0),
            fused_joint.data[0, 0].mean(0),
        ]
    )

    border_ids = np.array(
        [
            tile_size - overlap,
            tile_size,
            tile_size - overlap - 1,
            tile_size - 1,
        ]
    )
    overlap_region_ids = np.arange(tile_size - overlap + 5, tile_size - 5)

    border_diffs = np.abs(profiles[0][border_ids] - profiles[1][border_ids])
    mean_overlap_diff = np.mean(
        np.abs(profiles[0][overlap_region_ids] - profiles[1][overlap_region_ids])
    )

    assert np.all(border_diffs <= 2 * mean_overlap_diff), (
        "Border-localized seam detected: "
        f"border_diffs={border_diffs}, "
        f"mean_overlap_diff={mean_overlap_diff}"
    )
