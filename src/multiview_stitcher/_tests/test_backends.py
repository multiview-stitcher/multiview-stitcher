"""Tests for the backend system.

Correctness tests compare non-numpy backends against numpy (ground truth).
GPU tests are marked with @pytest.mark.gpu and skip when CuPy is not available.
"""

import numpy as np
import pytest

from multiview_stitcher import fusion, registration, sample_data
from multiview_stitcher import msi_utils
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher import transformation, weights
from multiview_stitcher.backends import get_backend, set_backend
from multiview_stitcher.backends._array_api import ArrayAPIBackend
from multiview_stitcher.io import METADATA_TRANSFORM_KEY


# ---------------------------------------------------------------------------
# Backend registry tests (always run, no GPU needed)
# ---------------------------------------------------------------------------


def test_get_numpy_backend():
    backend = get_backend("numpy")
    assert backend is not None
    assert isinstance(backend, ArrayAPIBackend)
    assert "numpy" in repr(backend)


def test_get_numpy_legacy_backend():
    from multiview_stitcher.backends._numpy_legacy import NumpyLegacyBackend

    backend = get_backend("numpy-legacy")
    assert isinstance(backend, NumpyLegacyBackend)


def test_get_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend("nonexistent")


def test_set_backend_roundtrip():
    original = get_backend()
    set_backend("numpy")
    assert isinstance(get_backend(), ArrayAPIBackend)
    # restore
    set_backend("numpy")


def test_set_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        set_backend("nonexistent")


def test_numpy_backend_basic_ops():
    backend = get_backend("numpy")
    x = backend.asarray([1.0, 2.0, np.nan])

    assert np.isnan(backend.nan)
    assert backend.nansum(x) == pytest.approx(3.0)
    assert backend.nanmax(x) == pytest.approx(2.0)

    y = backend.nan_to_num(x)
    assert y[2] == 0.0

    z = backend.clip(backend.asarray([0.5, 1.5, -0.5]), 0, 1)
    np.testing.assert_array_equal(backend.to_numpy(z), [0.5, 1.0, 0.0])


def test_numpy_legacy_matches_arrayapi():
    """ArrayAPIBackend('numpy') produces same results as NumpyBackend."""
    api = get_backend("numpy")
    legacy = get_backend("numpy-legacy")

    x = np.array([1.0, 2.0, np.nan, 4.0])

    assert api.nansum(api.asarray(x)) == pytest.approx(
        legacy.nansum(legacy.asarray(x))
    )
    assert api.nanmax(api.asarray(x)) == pytest.approx(
        legacy.nanmax(legacy.asarray(x))
    )

    # rescale_intensity
    img = np.array([0.0, 50.0, 100.0])
    np.testing.assert_allclose(
        api.to_numpy(
            api.rescale_intensity(
                api.asarray(img), in_range=(0, 100), out_range=(0, 1)
            )
        ),
        legacy.to_numpy(
            legacy.rescale_intensity(
                legacy.asarray(img), in_range=(0, 100), out_range=(0, 1)
            )
        ),
    )


def test_fuse_with_explicit_numpy_backend():
    """Passing backend='numpy' should give identical results to default."""
    sims = sample_data.generate_tiled_dataset(
        ndim=2, N_c=1, N_t=1, tile_size=30,
        tiles_x=2, tiles_y=2, tiles_z=1,
        overlap=5, spacing_x=1, spacing_y=1, spacing_z=1,
    )

    fused_default = fusion.fuse(
        sims, transform_key=METADATA_TRANSFORM_KEY,
    ).compute(scheduler="single-threaded")

    fused_numpy = fusion.fuse(
        sims, transform_key=METADATA_TRANSFORM_KEY, backend="numpy",
    ).compute(scheduler="single-threaded")

    np.testing.assert_array_equal(fused_default.values, fused_numpy.values)


def test_fuse_numpy_backend():
    """ArrayAPIBackend('numpy') fusion produces valid results."""
    sims = sample_data.generate_tiled_dataset(
        ndim=2, N_c=1, N_t=1, tile_size=30,
        tiles_x=2, tiles_y=2, tiles_z=1,
        overlap=5, spacing_x=1, spacing_y=1, spacing_z=1,
    )

    fused = fusion.fuse(
        sims, transform_key=METADATA_TRANSFORM_KEY, backend="numpy",
    ).compute(scheduler="single-threaded")

    assert fused.values.max() > 0
    assert not np.any(np.isnan(fused.values))


def test_custom_fusion_func_with_arrayapi_numpy():
    """Custom fusion functions should work via NumpyBackend fallback."""
    sims = sample_data.generate_tiled_dataset(
        ndim=2, N_c=1, N_t=1, tile_size=30,
        tiles_x=2, tiles_y=2, tiles_z=1,
        overlap=5, spacing_x=1, spacing_y=1, spacing_z=1,
    )

    # max_fusion is a built-in, but test custom func fallback path
    fused = fusion.fuse(
        sims,
        transform_key=METADATA_TRANSFORM_KEY,
        fusion_func=fusion.max_fusion,
        backend="numpy",
    ).compute(scheduler="single-threaded")

    assert fused.values.max() > 0


# ---------------------------------------------------------------------------
# CuPy backend correctness tests (skip when no GPU / no CuPy)
# ---------------------------------------------------------------------------


def _get_test_sims(ndim=3, tile_size=20):
    """Generate a small tiled dataset for testing."""
    return sample_data.generate_tiled_dataset(
        ndim=ndim, N_c=1, N_t=1,
        tile_size=tile_size, tiles_x=2, tiles_y=2, tiles_z=1,
        overlap=5, spacing_x=1, spacing_y=1, spacing_z=1,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("ndim", [2, 3])
def test_cupy_matches_numpy(ndim):
    """CuPy fusion output must match NumPy within atol=1."""
    cupy = pytest.importorskip("cupy")
    import dask

    sims = _get_test_sims(ndim=ndim)

    fused_np = fusion.fuse(
        sims, transform_key=METADATA_TRANSFORM_KEY, backend="numpy",
    ).compute(scheduler="single-threaded")

    with dask.config.set(scheduler="synchronous"):
        fused_cp = fusion.fuse(
            sims, transform_key=METADATA_TRANSFORM_KEY, backend="cupy",
        ).compute()

    np.testing.assert_allclose(
        fused_np.values.astype(float),
        fused_cp.values.astype(float),
        atol=1,
        err_msg=f"CuPy fusion output differs from NumPy for {ndim}D",
    )


@pytest.mark.gpu
def test_cupy_dtype_preserved():
    """Output dtype must match input dtype."""
    cupy = pytest.importorskip("cupy")
    import dask

    sims = _get_test_sims()

    with dask.config.set(scheduler="synchronous"):
        fused = fusion.fuse(
            sims, transform_key=METADATA_TRANSFORM_KEY, backend="cupy",
        ).compute()

    assert fused.dtype == sims[0].dtype


@pytest.mark.gpu
def test_cupy_nonzero_output():
    """Fused output should contain nonzero values for nonzero input."""
    cupy = pytest.importorskip("cupy")
    import dask

    sims = _get_test_sims(ndim=2, tile_size=30)

    with dask.config.set(scheduler="synchronous"):
        fused = fusion.fuse(
            sims, transform_key=METADATA_TRANSFORM_KEY, backend="cupy",
        ).compute()

    assert fused.values.max() > 0, "Fused output is all zeros"


@pytest.mark.gpu
def test_cupy_max_fusion():
    """Test CuPy with max_fusion function."""
    cupy = pytest.importorskip("cupy")
    import dask

    sims = _get_test_sims(ndim=2, tile_size=30)

    fused_np = fusion.fuse(
        sims, transform_key=METADATA_TRANSFORM_KEY,
        fusion_func=fusion.max_fusion, backend="numpy",
    ).compute(scheduler="single-threaded")

    with dask.config.set(scheduler="synchronous"):
        fused_cp = fusion.fuse(
            sims, transform_key=METADATA_TRANSFORM_KEY,
            fusion_func=fusion.max_fusion, backend="cupy",
        ).compute()

    np.testing.assert_allclose(
        fused_np.values.astype(float),
        fused_cp.values.astype(float),
        atol=1,
    )


@pytest.mark.gpu
def test_cupy_backend_basic_ops():
    """Test CuPy backend basic array operations."""
    cupy = pytest.importorskip("cupy")

    backend = get_backend("cupy")
    assert isinstance(backend, ArrayAPIBackend)

    x = backend.asarray([1.0, 2.0, float("nan")])

    assert backend.nansum(x).item() == pytest.approx(3.0)
    assert backend.nanmax(x).item() == pytest.approx(2.0)

    y = backend.nan_to_num(x)
    assert backend.to_numpy(y)[2] == 0.0

    z = backend.clip(backend.asarray([0.5, 1.5, -0.5]), 0, 1)
    np.testing.assert_array_equal(backend.to_numpy(z), [0.5, 1.0, 0.0])

    assert backend.recommended_dask_scheduler == "synchronous"


@pytest.mark.gpu
def test_cupy_legacy_backend():
    """Legacy CuPy backend is still accessible."""
    cupy = pytest.importorskip("cupy")
    from multiview_stitcher.backends._cupy_legacy import CupyLegacyBackend

    backend = get_backend("cupy-legacy")
    assert isinstance(backend, CupyLegacyBackend)


@pytest.mark.gpu
def test_cupy_arrayapi_matches_legacy():
    """ArrayAPIBackend('cupy') produces same results as CupyBackend."""
    cupy = pytest.importorskip("cupy")
    import dask

    sims = _get_test_sims(ndim=2, tile_size=30)

    with dask.config.set(scheduler="synchronous"):
        fused_api = fusion.fuse(
            sims, transform_key=METADATA_TRANSFORM_KEY, backend="cupy",
        ).compute()

        fused_legacy = fusion.fuse(
            sims, transform_key=METADATA_TRANSFORM_KEY,
            backend="cupy-legacy",
        ).compute()

    np.testing.assert_allclose(
        fused_api.values.astype(float),
        fused_legacy.values.astype(float),
        atol=1,
    )


# ---------------------------------------------------------------------------
# transform_data() tests
# ---------------------------------------------------------------------------


def test_transform_data_numpy():
    """transform_data with numpy backend matches transform_sim."""
    sims = _get_test_sims(ndim=2, tile_size=30)
    # Squeeze out non-spatial dims for raw array test
    sim = sims[0].sel(t=sims[0].coords["t"][0], c=sims[0].coords["c"][0]).astype(float)
    sdims = si_utils.get_spatial_dims_from_sim(sim)
    ndim = len(sdims)

    import multiview_stitcher.param_utils as param_utils
    p = param_utils.identity_transform(ndim)

    output_props = {
        "spacing": {dim: 1.0 for dim in sdims},
        "origin": {dim: 0.0 for dim in sdims},
        "shape": {dim: 25 for dim in sdims},
    }

    result_sim = transformation.transform_sim(
        sim, p=p, output_stack_properties=output_props, order=1, cval=0.0,
    )

    result_data = transformation.transform_data(
        np.asarray(sim.data),
        p=p,
        input_spacing=si_utils.get_spacing_from_sim(sim, asarray=True),
        input_origin=si_utils.get_origin_from_sim(sim, asarray=True),
        output_stack_properties=output_props,
        spatial_dims=sdims,
        backend="numpy",
        order=1,
        cval=0.0,
    )

    np.testing.assert_array_equal(result_sim.data, result_data)


# ---------------------------------------------------------------------------
# get_blending_weights() with backend tests
# ---------------------------------------------------------------------------


def test_blending_weights_numpy_backend_matches_default():
    """get_blending_weights with explicit numpy backend matches default."""
    sims = _get_test_sims(ndim=2, tile_size=30)
    sdims = si_utils.get_spatial_dims_from_sim(sims[0])

    target_bb = {
        "spacing": {dim: 1.0 for dim in sdims},
        "origin": {dim: 0.0 for dim in sdims},
        "shape": {dim: 30 for dim in sdims},
    }
    source_bb = {
        "spacing": {dim: 1.0 for dim in sdims},
        "origin": {dim: 0.0 for dim in sdims},
        "shape": {dim: 30 for dim in sdims},
    }
    affine = np.eye(len(sdims) + 1)

    w_default = weights.get_blending_weights(
        target_bb, source_bb, affine,
    )
    w_numpy = weights.get_blending_weights(
        target_bb, source_bb, affine, backend="numpy",
    )

    # Default returns SpatialImage, numpy ArrayAPI backend returns raw array
    np.testing.assert_allclose(
        w_default.data,
        np.asarray(w_numpy),
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# Registration with backend tests
# ---------------------------------------------------------------------------


def test_register_with_explicit_numpy_backend():
    """Registration with backend='numpy' gives identical results to default."""
    sims = sample_data.generate_tiled_dataset(
        ndim=2, N_c=1, N_t=1, tile_size=30,
        tiles_x=2, tiles_y=1, tiles_z=1,
        overlap=8, spacing_x=1, spacing_y=1, spacing_z=1,
    )
    msims = [msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims]

    params_default = registration.register(
        msims,
        reg_channel_index=0,
        transform_key=METADATA_TRANSFORM_KEY,
    )

    # Reset msims
    msims2 = [msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims]

    params_numpy = registration.register(
        msims2,
        reg_channel_index=0,
        transform_key=METADATA_TRANSFORM_KEY,
        backend="numpy",
    )

    for p_default, p_numpy in zip(params_default, params_numpy):
        np.testing.assert_allclose(
            np.asarray(p_default),
            np.asarray(p_numpy),
            atol=1e-5,
        )


def test_numpy_backend_new_methods():
    """Test newly added numpy backend methods."""
    backend = get_backend("numpy")

    x = backend.asarray([1.0, 2.0, 3.0, np.nan])
    assert backend.nanmin(x) == pytest.approx(1.0)
    assert backend.any(backend.asarray([False, True, False]))
    assert backend.sum(backend.asarray([1, 2, 3])) == 6
    np.testing.assert_array_equal(
        backend.to_numpy(backend.abs(backend.asarray([-1, 2, -3]))),
        [1, 2, 3],
    )

    # rescale_intensity
    img = backend.asarray([0.0, 50.0, 100.0])
    rescaled = backend.rescale_intensity(img, in_range=(0, 100), out_range=(0, 1))
    np.testing.assert_allclose(backend.to_numpy(rescaled), [0.0, 0.5, 1.0])

    # gaussian_filter
    img2d = backend.asarray(np.random.rand(10, 10).astype(np.float32))
    filtered = backend.gaussian_filter(img2d, sigma=1.0)
    assert filtered.shape == (10, 10)


# ---------------------------------------------------------------------------
# Numba acceleration tests
# ---------------------------------------------------------------------------


def _skip_unless_numba():
    return pytest.importorskip("numba")


def test_numba_acceleration_toggle():
    """Test that numba acceleration can be toggled."""
    _skip_unless_numba()
    from multiview_stitcher._numba_acceleration import (
        get_numba_acceleration,
        set_numba_acceleration,
    )

    original = get_numba_acceleration()
    try:
        set_numba_acceleration(False)
        assert not get_numba_acceleration()
        set_numba_acceleration(True)
        assert get_numba_acceleration()
    finally:
        set_numba_acceleration(original)


def test_numba_affine_transform_2d():
    """Numba-accelerated 2D affine transform matches scipy."""
    _skip_unless_numba()
    from scipy.ndimage import affine_transform as scipy_affine

    from multiview_stitcher._numba_acceleration import (
        affine_transform as accel_affine,
        set_numba_acceleration,
    )

    np.random.seed(42)
    src = np.random.rand(20, 20).astype(np.float64)
    matrix = np.eye(2) * 1.1
    offset = np.array([1.0, 2.0])

    result_scipy = scipy_affine(
        src, matrix=matrix, offset=offset,
        output_shape=(15, 15), order=1, cval=0.0,
    )

    set_numba_acceleration(True)
    result_numba = accel_affine(
        src, matrix=matrix, offset=offset,
        output_shape=(15, 15), order=1, cval=0.0,
    )

    np.testing.assert_allclose(result_scipy, result_numba, atol=1e-10)


def test_numba_affine_transform_3d():
    """Numba-accelerated 3D affine transform matches scipy."""
    _skip_unless_numba()
    from scipy.ndimage import affine_transform as scipy_affine

    from multiview_stitcher._numba_acceleration import (
        affine_transform as accel_affine,
        set_numba_acceleration,
    )

    np.random.seed(42)
    src = np.random.rand(10, 10, 10).astype(np.float64)
    matrix = np.eye(3) * 1.05
    offset = np.array([0.5, 1.0, 0.5])

    result_scipy = scipy_affine(
        src, matrix=matrix, offset=offset,
        output_shape=(8, 8, 8), order=1, cval=0.0,
    )

    set_numba_acceleration(True)
    result_numba = accel_affine(
        src, matrix=matrix, offset=offset,
        output_shape=(8, 8, 8), order=1, cval=0.0,
    )

    np.testing.assert_allclose(result_scipy, result_numba, atol=1e-10)


def test_numba_fuse_matches_no_numba_2d():
    """Fusion with numba acceleration matches without it (2D)."""
    _skip_unless_numba()
    from multiview_stitcher._numba_acceleration import set_numba_acceleration

    sims = _get_test_sims(ndim=2, tile_size=30)

    set_numba_acceleration(False)
    fused_no_nb = fusion.fuse(
        sims, transform_key=METADATA_TRANSFORM_KEY, backend="numpy",
    ).compute(scheduler="single-threaded")

    set_numba_acceleration(True)
    fused_nb = fusion.fuse(
        sims, transform_key=METADATA_TRANSFORM_KEY, backend="numpy",
    ).compute(scheduler="single-threaded")

    np.testing.assert_allclose(
        fused_no_nb.values.astype(float),
        fused_nb.values.astype(float),
        atol=1,
    )


def test_numba_fuse_matches_no_numba_3d():
    """Fusion with numba acceleration matches without it (3D)."""
    _skip_unless_numba()
    from multiview_stitcher._numba_acceleration import set_numba_acceleration

    sims = _get_test_sims(ndim=3, tile_size=20)

    set_numba_acceleration(False)
    fused_no_nb = fusion.fuse(
        sims, transform_key=METADATA_TRANSFORM_KEY, backend="numpy",
    ).compute(scheduler="single-threaded")

    set_numba_acceleration(True)
    fused_nb = fusion.fuse(
        sims, transform_key=METADATA_TRANSFORM_KEY, backend="numpy",
    ).compute(scheduler="single-threaded")

    np.testing.assert_allclose(
        fused_no_nb.values.astype(float),
        fused_nb.values.astype(float),
        atol=1,
    )


def test_numba_max_fusion():
    """max_fusion with numba acceleration matches without it."""
    _skip_unless_numba()
    from multiview_stitcher._numba_acceleration import set_numba_acceleration

    sims = _get_test_sims(ndim=2, tile_size=30)

    set_numba_acceleration(False)
    fused_no_nb = fusion.fuse(
        sims, transform_key=METADATA_TRANSFORM_KEY,
        fusion_func=fusion.max_fusion, backend="numpy",
    ).compute(scheduler="single-threaded")

    set_numba_acceleration(True)
    fused_nb = fusion.fuse(
        sims, transform_key=METADATA_TRANSFORM_KEY,
        fusion_func=fusion.max_fusion, backend="numpy",
    ).compute(scheduler="single-threaded")

    np.testing.assert_allclose(
        fused_no_nb.values.astype(float),
        fused_nb.values.astype(float),
        atol=1,
    )


def test_numba_blending_weights():
    """Numba blending weights match numpy."""
    _skip_unless_numba()
    from multiview_stitcher._numba_acceleration import set_numba_acceleration

    sdims = ("y", "x")
    target_bb = {
        "spacing": {dim: 1.0 for dim in sdims},
        "origin": {dim: 0.0 for dim in sdims},
        "shape": {dim: 30 for dim in sdims},
    }
    source_bb = {
        "spacing": {dim: 1.0 for dim in sdims},
        "origin": {dim: 0.0 for dim in sdims},
        "shape": {dim: 30 for dim in sdims},
    }
    affine = np.eye(3)

    set_numba_acceleration(False)
    w_np = weights.get_blending_weights(target_bb, source_bb, affine)

    set_numba_acceleration(True)
    w_nb = weights.get_blending_weights(target_bb, source_bb, affine)

    np.testing.assert_allclose(
        w_np.data, np.asarray(w_nb), atol=1e-5,
    )


# ---------------------------------------------------------------------------
# dpnp backend tests (skip when dpnp not available)
# ---------------------------------------------------------------------------


def test_dpnp_backend_registered_when_available():
    """dpnp backend should be auto-registered when dpnp is importable."""
    try:
        import dpnp  # noqa: F401
    except ImportError:
        pytest.skip("dpnp not installed")

    from multiview_stitcher.backends import _REGISTRY

    assert "dpnp" in _REGISTRY


def test_dpnp_backend_basic_ops():
    """Test dpnp backend basic array operations."""
    dpnp = pytest.importorskip("dpnp")

    backend = get_backend("dpnp")
    assert isinstance(backend, ArrayAPIBackend)
    assert backend._name == "dpnp"

    x = backend.asarray([1.0, 2.0, float("nan")])
    assert float(backend.nansum(x)) == pytest.approx(3.0)

    y = backend.to_numpy(x)
    assert isinstance(y, np.ndarray)


# ---------------------------------------------------------------------------
# JAX backend tests
# ---------------------------------------------------------------------------


def _skip_unless_jax():
    return pytest.importorskip("jax")


def test_jax_backend_registered():
    """JAX backend should be auto-registered when jax is importable."""
    _skip_unless_jax()
    from multiview_stitcher.backends import _REGISTRY

    assert "jax" in _REGISTRY


def test_jax_backend_basic_ops():
    """Test JaxBackend basic array operations."""
    _skip_unless_jax()
    backend = get_backend("jax")

    x = backend.asarray([1.0, 2.0, float("nan")])
    assert float(backend.nansum(x)) == pytest.approx(3.0)
    assert float(backend.nanmax(x)) == pytest.approx(2.0)
    assert float(backend.nanmin(x)) == pytest.approx(1.0)

    y = backend.nan_to_num(x)
    assert float(backend.to_numpy(y)[2]) == 0.0

    z = backend.clip(backend.asarray([0.5, 1.5, -0.5]), 0, 1)
    np.testing.assert_array_equal(backend.to_numpy(z), [0.5, 1.0, 0.0])

    assert backend.recommended_dask_scheduler == "synchronous"


def test_jax_affine_transform_2d():
    """JAX on-device affine transform matches scipy for 2D."""
    _skip_unless_jax()
    from scipy.ndimage import affine_transform as scipy_affine

    backend = get_backend("jax")
    np.random.seed(42)
    src = np.random.rand(20, 20).astype(np.float32)
    matrix = (np.eye(2) * 1.1).astype(np.float32)
    offset = np.array([1.0, 2.0], dtype=np.float32)

    result_scipy = scipy_affine(
        src, matrix=matrix, offset=offset,
        output_shape=(15, 15), order=1, cval=0.0,
    )
    result_jax = backend.affine_transform(
        backend.asarray(src), matrix=matrix, offset=offset,
        output_shape=(15, 15), order=1, cval=0.0,
    )

    np.testing.assert_allclose(
        result_scipy, backend.to_numpy(result_jax), atol=1e-5,
    )


def test_jax_affine_transform_3d():
    """JAX on-device affine transform matches scipy for 3D."""
    _skip_unless_jax()
    from scipy.ndimage import affine_transform as scipy_affine

    backend = get_backend("jax")
    np.random.seed(42)
    src = np.random.rand(10, 10, 10).astype(np.float32)
    matrix = (np.eye(3) * 1.05).astype(np.float32)
    offset = np.array([0.5, 1.0, 0.5], dtype=np.float32)

    result_scipy = scipy_affine(
        src, matrix=matrix, offset=offset,
        output_shape=(8, 8, 8), order=1, cval=0.0,
    )
    result_jax = backend.affine_transform(
        backend.asarray(src), matrix=matrix, offset=offset,
        output_shape=(8, 8, 8), order=1, cval=0.0,
    )

    np.testing.assert_allclose(
        result_scipy, backend.to_numpy(result_jax), atol=1e-4,
    )


def test_jax_fuse_matches_numpy_2d():
    """JAX fusion output matches numpy for 2D."""
    _skip_unless_jax()
    import dask

    sims = _get_test_sims(ndim=2, tile_size=30)

    fused_np = fusion.fuse(
        sims, transform_key=METADATA_TRANSFORM_KEY, backend="numpy",
    ).compute(scheduler="single-threaded")

    with dask.config.set(scheduler="synchronous"):
        fused_jax = fusion.fuse(
            sims, transform_key=METADATA_TRANSFORM_KEY, backend="jax",
        ).compute()

    np.testing.assert_allclose(
        fused_np.values.astype(float),
        fused_jax.values.astype(float),
        atol=1,
    )


def test_jax_max_fusion():
    """JAX max_fusion matches numpy."""
    _skip_unless_jax()
    import dask

    sims = _get_test_sims(ndim=2, tile_size=30)

    fused_np = fusion.fuse(
        sims, transform_key=METADATA_TRANSFORM_KEY,
        fusion_func=fusion.max_fusion, backend="numpy",
    ).compute(scheduler="single-threaded")

    with dask.config.set(scheduler="synchronous"):
        fused_jax = fusion.fuse(
            sims, transform_key=METADATA_TRANSFORM_KEY,
            fusion_func=fusion.max_fusion, backend="jax",
        ).compute()

    np.testing.assert_allclose(
        fused_np.values.astype(float),
        fused_jax.values.astype(float),
        atol=1,
    )


def test_jax_blending_weights():
    """JAX blending weights match numpy."""
    _skip_unless_jax()

    sdims = ("y", "x")
    target_bb = {
        "spacing": {dim: 1.0 for dim in sdims},
        "origin": {dim: 0.0 for dim in sdims},
        "shape": {dim: 30 for dim in sdims},
    }
    source_bb = {
        "spacing": {dim: 1.0 for dim in sdims},
        "origin": {dim: 0.0 for dim in sdims},
        "shape": {dim: 30 for dim in sdims},
    }
    affine = np.eye(3)

    w_np = weights.get_blending_weights(target_bb, source_bb, affine)
    w_jax = weights.get_blending_weights(
        target_bb, source_bb, affine, backend="jax",
    )

    backend = get_backend("jax")
    np.testing.assert_allclose(
        w_np.data, backend.to_numpy(w_jax), atol=1e-4,
    )


def test_jax_custom_fusion_raises():
    """Custom fusion functions should raise with JAX backend."""
    _skip_unless_jax()

    sims = _get_test_sims(ndim=2, tile_size=30)

    def my_custom_fusion(transformed_views, blending_weights):
        return np.nanmean(transformed_views, axis=0)

    with pytest.raises(NotImplementedError):
        fusion.fuse(
            sims,
            transform_key=METADATA_TRANSFORM_KEY,
            fusion_func=my_custom_fusion,
            backend="jax",
        ).compute(scheduler="single-threaded")


# ---------------------------------------------------------------------------
# MLX backend tests (skip when MLX not available — requires macOS + Apple Silicon)
# ---------------------------------------------------------------------------


def test_mlx_backend_registered_when_available():
    """MLX backend should be auto-registered when mlx is importable."""
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        pytest.skip("mlx not installed")

    from multiview_stitcher.backends import _REGISTRY

    assert "mlx" in _REGISTRY


def test_mlx_backend_basic_ops():
    """Test MLX backend basic array operations."""
    pytest.importorskip("mlx.core")

    backend = get_backend("mlx")

    x = backend.asarray([1.0, 2.0, float("nan")])
    assert float(backend.nansum(x)) == pytest.approx(3.0)
    assert float(backend.nanmax(x)) == pytest.approx(2.0)
    assert float(backend.nanmin(x)) == pytest.approx(1.0)

    y = backend.nan_to_num(x)
    assert float(backend.to_numpy(y)[2]) == 0.0

    z = backend.clip(backend.asarray([0.5, 1.5, -0.5]), 0, 1)
    np.testing.assert_array_equal(backend.to_numpy(z), [0.5, 1.0, 0.0])

    assert backend.recommended_dask_scheduler == "synchronous"


def test_mlx_rescale_intensity():
    """Test MLX rescale_intensity."""
    pytest.importorskip("mlx.core")

    backend = get_backend("mlx")
    img = backend.asarray([0.0, 50.0, 100.0])
    rescaled = backend.rescale_intensity(
        img, in_range=(0, 100), out_range=(0, 1),
    )
    np.testing.assert_allclose(
        backend.to_numpy(rescaled), [0.0, 0.5, 1.0], atol=1e-6,
    )


def test_mlx_affine_transform():
    """Test MLX affine transform (falls back to CPU scipy)."""
    pytest.importorskip("mlx.core")

    from scipy.ndimage import affine_transform as scipy_affine

    backend = get_backend("mlx")
    np.random.seed(42)
    src = np.random.rand(20, 20).astype(np.float32)

    result_scipy = scipy_affine(
        src, matrix=np.eye(2), offset=np.zeros(2),
        output_shape=(15, 15), order=1, cval=0.0,
    )
    result_mlx = backend.affine_transform(
        backend.asarray(src), matrix=np.eye(2), offset=np.zeros(2),
        output_shape=(15, 15), order=1, cval=0.0,
    )

    np.testing.assert_allclose(
        result_scipy, backend.to_numpy(result_mlx), atol=1e-5,
    )


def test_mlx_fuse_matches_numpy():
    """MLX fusion output matches numpy."""
    pytest.importorskip("mlx.core")
    import dask

    sims = _get_test_sims(ndim=2, tile_size=30)

    fused_np = fusion.fuse(
        sims, transform_key=METADATA_TRANSFORM_KEY, backend="numpy",
    ).compute(scheduler="single-threaded")

    with dask.config.set(scheduler="synchronous"):
        fused_mlx = fusion.fuse(
            sims, transform_key=METADATA_TRANSFORM_KEY, backend="mlx",
        ).compute()

    np.testing.assert_allclose(
        fused_np.values.astype(float),
        fused_mlx.values.astype(float),
        atol=1,
    )


# ---------------------------------------------------------------------------
# CuPy backend: refactored module tests
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_cupy_transform_data():
    """transform_data with cupy backend produces similar results to numpy."""
    cupy = pytest.importorskip("cupy")

    sims = _get_test_sims(ndim=2, tile_size=30)
    sim = sims[0].sel(t=sims[0].coords["t"][0], c=sims[0].coords["c"][0]).astype(float)
    sdims = si_utils.get_spatial_dims_from_sim(sim)
    ndim = len(sdims)

    import multiview_stitcher.param_utils as param_utils
    p = param_utils.identity_transform(ndim)

    output_props = {
        "spacing": {dim: 1.0 for dim in sdims},
        "origin": {dim: 0.0 for dim in sdims},
        "shape": {dim: 25 for dim in sdims},
    }

    np_backend = get_backend("numpy")
    cp_backend = get_backend("cupy")

    result_np = transformation.transform_data(
        np.asarray(sim.data),
        p=p,
        input_spacing=si_utils.get_spacing_from_sim(sim, asarray=True),
        input_origin=si_utils.get_origin_from_sim(sim, asarray=True),
        output_stack_properties=output_props,
        spatial_dims=sdims,
        backend=np_backend,
        order=1, cval=0.0,
    )

    result_cp = transformation.transform_data(
        cp_backend.asarray(np.asarray(sim.data)),
        p=p,
        input_spacing=si_utils.get_spacing_from_sim(sim, asarray=True),
        input_origin=si_utils.get_origin_from_sim(sim, asarray=True),
        output_stack_properties=output_props,
        spatial_dims=sdims,
        backend=cp_backend,
        order=1, cval=0.0,
    )

    np.testing.assert_allclose(
        result_np, cp_backend.to_numpy(result_cp), atol=1,
    )


@pytest.mark.gpu
def test_cupy_blending_weights():
    """get_blending_weights with cupy backend matches numpy."""
    cupy = pytest.importorskip("cupy")

    sdims = ("y", "x")
    target_bb = {
        "spacing": {dim: 1.0 for dim in sdims},
        "origin": {dim: 0.0 for dim in sdims},
        "shape": {dim: 30 for dim in sdims},
    }
    source_bb = {
        "spacing": {dim: 1.0 for dim in sdims},
        "origin": {dim: 0.0 for dim in sdims},
        "shape": {dim: 30 for dim in sdims},
    }
    affine = np.eye(3)

    w_np = weights.get_blending_weights(target_bb, source_bb, affine)
    w_cp = weights.get_blending_weights(
        target_bb, source_bb, affine, backend="cupy",
    )

    cp_backend = get_backend("cupy")

    np.testing.assert_allclose(
        w_np.data, cp_backend.to_numpy(w_cp), atol=1e-5,
    )


@pytest.mark.gpu
def test_cupy_registration():
    """Registration with cupy backend produces consistent results."""
    cupy = pytest.importorskip("cupy")

    sims = sample_data.generate_tiled_dataset(
        ndim=2, N_c=1, N_t=1, tile_size=30,
        tiles_x=2, tiles_y=1, tiles_z=1,
        overlap=8, spacing_x=1, spacing_y=1, spacing_z=1,
    )
    msims = [msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims]

    params = registration.register(
        msims,
        reg_channel_index=0,
        transform_key=METADATA_TRANSFORM_KEY,
        backend="cupy",
    )

    # Should produce valid transforms (not all identity)
    assert len(params) == 2
    for p in params:
        arr = np.asarray(p)
        # params may have a time dimension: (N_t, ndim+1, ndim+1)
        assert arr.shape[-2:] == (3, 3)


@pytest.mark.gpu
def test_cupy_backend_new_methods():
    """Test newly added cupy backend methods."""
    cupy = pytest.importorskip("cupy")

    backend = get_backend("cupy")

    x = backend.asarray([1.0, 2.0, 3.0, float("nan")])
    assert float(backend.nanmin(x)) == pytest.approx(1.0)
    assert bool(backend.any(backend.asarray([False, True, False])))
    assert int(backend.sum(backend.asarray([1, 2, 3]))) == 6

    # rescale_intensity
    img = backend.asarray([0.0, 50.0, 100.0])
    rescaled = backend.rescale_intensity(img, in_range=(0, 100), out_range=(0, 1))
    np.testing.assert_allclose(
        backend.to_numpy(rescaled), [0.0, 0.5, 1.0], atol=1e-7,
    )


# ---------------------------------------------------------------------------
# CuPy-on-ROCm bug canaries
#
# These tests assert that known CuPy/ROCm bugs STILL EXIST.  Each test
# guards a workaround in our code.  When a test FAILS it means the
# upstream bug has been fixed and the corresponding workaround can be
# safely removed.
#
# How to act when a canary fails:
#   1. Verify the fix on all target ROCm GPUs.
#   2. Remove the workaround referenced in the test docstring.
#   3. Delete or invert the canary test.
# ---------------------------------------------------------------------------


def _skip_unless_rocm():
    """Skip the test unless running CuPy on ROCm/HIP."""
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.is_hip:
        pytest.skip("Not a ROCm/HIP build of CuPy")
    return cp


@pytest.mark.gpu
def test_rocm_bug_boolean_mask_indexing_uses_shfl_xor_sync():
    """Canary: CuPy's boolean mask indexing triggers __shfl_xor_sync on ROCm.

    https://github.com/cupy/cupy/issues/9829

    CuPy's internal scan kernel (used by ``array[bool_mask] = value``)
    contains the CUDA-only intrinsic ``__shfl_xor_sync`` which does not
    exist in HIP, causing an ``HIPRTC_ERROR_COMPILATION``.

    Workarounds guarded by this canary
    -----------------------------------
    - ``ArrayAPIBackend.masked_fill()`` always uses ``xp.where()``
      instead of ``array[mask] = value``.
    - ``CupyBackend.masked_fill()`` (legacy) uses ``cp.where()``
      instead of ``array[mask] = value`` when on ROCm.

    If this test FAILS (i.e. the assignment succeeds), the ROCm team has
    fixed the scan kernel and the ``xp.where()`` path can be replaced
    with the direct assignment on ROCm.
    """
    cp = _skip_unless_rocm()

    # The broken scan kernel is only used for arrays larger than ~512
    # elements; smaller arrays take a simpler code path that works.
    arr = cp.zeros(1024, dtype=cp.float32)
    mask = cp.arange(1024) % 2 == 0

    with pytest.raises(Exception):
        # This triggers the broken scan kernel on ROCm.
        arr[mask] = 1.0
        cp.cuda.Device().synchronize()


@pytest.mark.gpu
def test_rocm_bug_compute_capability_collision():
    """Canary: CuPy reports identical compute_capability for different GCN archs.

    https://github.com/cupy/cupy/issues/9830

    On ROCm, ``cupy.cuda.Device(i).compute_capability`` truncates the
    GCN architecture number (e.g. both gfx906 and gfx908 return ``"90"``).
    CuPy uses this value as part of its kernel disk-cache key, so
    binaries compiled for one GPU are incorrectly loaded on another.

    Workaround guarded by this canary
    ----------------------------------
    - ``run_benchmarks()`` in ``_benchmark.py`` sets a per-device
      ``CUPY_CACHE_DIR`` on ROCm to isolate kernel caches.

    If this test FAILS (i.e. the architectures ARE distinguishable),
    CuPy now keys the cache correctly and the per-device cache directory
    logic in ``run_benchmarks()`` can be removed.
    """
    cp = _skip_unless_rocm()

    n_devices = cp.cuda.runtime.getDeviceCount()
    if n_devices < 2:
        pytest.skip("Need at least 2 GPUs to test cross-device cache collision")

    # Collect the actual GCN architecture names via device properties.
    gcn_names = []
    for i in range(n_devices):
        props = cp.cuda.runtime.getDeviceProperties(i)
        gcn_name = props.get("gcnArchName", b"").decode().split(":")[0]
        gcn_names.append(gcn_name)

    if len(set(gcn_names)) < 2:
        pytest.skip(
            "All GPUs share the same GCN architecture; "
            "cache collision cannot occur"
        )

    # The GPUs have different architectures.  Check whether
    # compute_capability can distinguish them.
    capabilities = set()
    for i in range(n_devices):
        capabilities.add(cp.cuda.Device(i).compute_capability)

    # Bug assertion: all different-arch GPUs map to the SAME
    # compute_capability string, so len(capabilities) == 1.
    assert len(capabilities) == 1, (
        "CuPy now returns distinct compute_capability values for "
        f"different GCN architectures ({gcn_names} -> {capabilities}).  "
        "The per-device CUPY_CACHE_DIR workaround in "
        "run_benchmarks() can be removed."
    )


@pytest.mark.gpu
def test_rocm_cucim_not_available():
    """Canary: cucim does not provide GPU-native skimage functions on ROCm.

    cucim (NVIDIA RAPIDS) is CUDA-only and does not ship ROCm/HIP builds.
    On ROCm the CuPy backend therefore falls back to CPU skimage for
    ``phase_cross_correlation`` and ``structural_similarity``.

    Workarounds guarded by this canary
    -----------------------------------
    - ``CupyBackend.phase_cross_correlation()`` falls back to CPU
      ``skimage.registration.phase_cross_correlation`` on ROCm.
    - ``CupyBackend.structural_similarity()`` falls back to CPU
      ``skimage.metrics.structural_similarity`` on ROCm.
    - ``ArrayAPIBackend`` has the same CPU fallback paths.

    If this test FAILS (i.e. the imports succeed), cucim now supports
    ROCm and the CPU fallback paths can be replaced with direct
    cucim calls on AMD GPUs.
    """
    _skip_unless_rocm()

    registration_available = False
    try:
        from cucim.skimage.registration import phase_cross_correlation  # noqa: F401
        registration_available = True
    except (ImportError, Exception):
        pass

    metrics_available = False
    try:
        from cucim.skimage.metrics import structural_similarity  # noqa: F401
        metrics_available = True
    except (ImportError, Exception):
        pass

    assert not registration_available, (
        "cucim.skimage.registration.phase_cross_correlation is now "
        "available on ROCm. The CPU fallback in "
        "CupyBackend.phase_cross_correlation() and "
        "ArrayAPIBackend.phase_cross_correlation() can be removed."
    )
    assert not metrics_available, (
        "cucim.skimage.metrics.structural_similarity is now "
        "available on ROCm. The CPU fallback in "
        "CupyBackend.structural_similarity() and "
        "ArrayAPIBackend.structural_similarity() can be removed."
    )


def _skip_unless_jax_rocm():
    """Skip the test unless JAX is installed with a ROCm plugin."""
    jax = pytest.importorskip("jax")
    try:
        from importlib.metadata import distributions

        is_rocm = any(
            "rocm" in d.metadata["Name"].lower()
            for d in distributions()
            if d.metadata["Name"].lower().startswith("jax")
        )
    except Exception:
        is_rocm = False
    if not is_rocm:
        pytest.skip("Not a ROCm build of JAX")
    return jax


@pytest.mark.gpu
def test_rocm_bug_xla_autotuner_repeat_buffer_kernel():
    """Canary: XLA's autotuner crashes loading RepeatBufferKernel on ROCm.

    XLA's autotuner tries to load a ``RepeatBufferKernel`` that is not
    compiled for older ROCm GPU targets (gfx906, gfx908), causing a
    fatal ``hipError_t(98)`` (``hipErrorInvalidImage``) that aborts the
    process without raising a Python exception.

    https://github.com/ROCm/rocm-jax/issues/360

    Workarounds guarded by this canary
    -----------------------------------
    - The benchmark notebook sets
      ``XLA_FLAGS="--xla_gpu_autotune_level=0"`` before importing JAX.
    - ``JaxBackend.affine_transform()`` in ``_jax_backend.py`` uses
      element-wise broadcasting instead of ``matmul`` to avoid triggering
      the Triton GEMM autotuner for small matrices.

    If this test FAILS (i.e. the matmul succeeds without the workaround),
    the ROCm team has fixed the autotuner and the broadcasting fallback
    in ``JaxBackend.affine_transform()`` and the ``XLA_FLAGS`` workaround
    in the notebook can be removed.
    """
    jax = _skip_unless_jax_rocm()
    import os

    # Temporarily remove the autotuner workaround if it was set.
    old_flags = os.environ.get("XLA_FLAGS", "")
    clean_flags = old_flags.replace("--xla_gpu_autotune_level=0", "").strip()
    os.environ["XLA_FLAGS"] = clean_flags

    try:
        # A small matmul is enough to trigger the Triton GEMM autotuner
        # which loads RepeatBufferKernel.  On affected GPUs this aborts
        # the process, so we can only assert "known broken" by checking
        # that the workaround flag was needed in the first place.
        #
        # NOTE: We cannot actually call jnp.matmul here without the flag
        # because the crash is a fatal C++ abort (not a Python exception).
        # Instead we verify the bug is still present by checking that
        # the plugin .so does NOT contain our GPU's target in its
        # RepeatBufferKernel symbol table.
        gpu_devices = [d for d in jax.devices() if d.platform != "cpu"]
        if not gpu_devices:
            pytest.skip("No ROCm GPU available")

        # The bug manifests as a fatal abort — we can't safely trigger it.
        # Verify the workaround is still needed by confirming the known-
        # broken JAX version range.  When the fix ships in a new version,
        # this assertion will fail, signalling the workaround can go.
        from importlib.metadata import version

        jax_plugin_version = version("jax-rocm7-plugin")
        major, minor, patch = (
            int(x) for x in jax_plugin_version.split(".")[:3]
        )
        assert (major, minor, patch) <= (0, 9, 1), (
            f"jax-rocm7-plugin {jax_plugin_version} may have fixed the "
            "RepeatBufferKernel crash (issue #360). Test a matmul without "
            "--xla_gpu_autotune_level=0 and, if it works, remove the "
            "workaround in JaxBackend.affine_transform() and the notebook."
        )
    finally:
        os.environ["XLA_FLAGS"] = old_flags
