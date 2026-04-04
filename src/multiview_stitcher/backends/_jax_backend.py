"""JAX backend for GPU / TPU / CPU acceleration.

Uses ``jax.numpy`` for array operations and
``jax.scipy.ndimage.map_coordinates`` for affine transforms (runs
entirely on-device -- no host round-trip).

Device support
--------------
* CPU  -- always available
* CUDA -- via ``jax[cuda12]``
* TPU  -- via Google Cloud / Colab
* Metal -- experimental, via ``jax-metal`` (no float64)

Float64
-------
JAX disables float64 by default.  Call
``jax.config.update("jax_enable_x64", True)`` before importing this
backend if you need it.  The backend works fine with float32 (the
default for the fusion pipeline).
"""

import numpy as np

from multiview_stitcher.backends._xp_backend import XPBackend

_jnp = None
_jsp_ndimage = None


def _ensure_jax():
    global _jnp, _jsp_ndimage
    if _jnp is None:
        import jax.numpy
        import jax.scipy.ndimage

        _jnp = jax.numpy
        _jsp_ndimage = jax.scipy.ndimage
    return _jnp, _jsp_ndimage


class JaxBackend(XPBackend):
    """JAX backend -- runs on CPU, CUDA GPU, or TPU.

    Affine transforms are implemented via
    ``jax.scipy.ndimage.map_coordinates`` so they execute entirely on
    the current JAX device.  Operations without a JAX equivalent
    (``distance_transform_edt``, ``gaussian_filter``, skimage metrics)
    fall back to CPU NumPy / SciPy.
    """

    _has_native_affine_transform = True
    _is_gpu = True  # JAX manages its own parallelism; synchronous avoids conflicts

    def __init__(self):
        jnp, self._ndimage = _ensure_jax()
        super().__init__(jnp, name="jax")
        self._is_rocm = self._detect_rocm()

    @staticmethod
    def _detect_rocm():
        """Return True when JAX is using a ROCm (AMD GPU) plugin."""
        try:
            from importlib.metadata import distributions
            return any(
                "rocm" in d.metadata["Name"].lower()
                for d in distributions()
                if d.metadata["Name"].lower().startswith("jax")
            )
        except Exception:
            return False

    # -- Overrides for JAX-specific behaviour -------------------------------

    def to_numpy(self, x):
        # np.asarray on a JAX array triggers device_get + block_until_ready
        return np.asarray(x)

    @property
    def nan(self):
        return float("nan")

    def masked_fill(self, array, mask, value):
        # JAX arrays are immutable -- no in-place mutation.
        return self.xp.where(mask, value, array)

    # -- Native affine transform via map_coordinates ------------------------

    def _native_affine_transform(
        self, input, matrix, offset, output_shape,
        mode="constant", cval=0.0, order=1,
    ):
        """On-device affine transform via ``map_coordinates``.

        Builds the source-coordinate grid on the JAX device and calls
        ``jax.scipy.ndimage.map_coordinates`` -- no host round-trip.
        """
        jnp = self.xp
        ndim = input.ndim

        matrix = jnp.asarray(matrix, dtype=input.dtype)
        offset = jnp.asarray(offset, dtype=input.dtype)

        # Build an (ndim, *output_shape) grid of output-pixel indices
        ranges = [jnp.arange(s, dtype=input.dtype) for s in output_shape]
        grid = jnp.meshgrid(*ranges, indexing="ij")
        coords = jnp.stack(grid)  # (ndim, *output_shape)

        # source_coord = matrix @ output_coord + offset
        flat = coords.reshape(ndim, -1)  # (ndim, N)
        if self._is_rocm:
            # ROCm workaround: XLA's Triton GEMM autotuner fails for small
            # matrices on AMD GPUs.  Use broadcasting instead of matmul.
            # https://github.com/ROCm/rocm-jax/issues/360
            # Canary: test_rocm_bug_xla_autotuner_repeat_buffer_kernel
            src = jnp.sum(
                matrix[:, :, None] * flat[None, :, :], axis=1,
            ) + offset[:, None]
        else:
            src = matrix @ flat + offset[:, None]  # (ndim, N)
        src = src.reshape(ndim, *output_shape)

        if mode == "constant":
            # JAX map_coordinates has two boundary quirks vs scipy:
            #  1. It clamps out-of-bounds coords instead of returning cval.
            #  2. With cval=NaN, NaN poisons linear interpolation at the
            #     last valid index (e.g. coord 29.0 for size 30) because
            #     JAX reads the next pixel (index 30 -> NaN) even though
            #     its interpolation weight is 0.
            # Fix: always interpolate with cval=0 (safe), then apply the
            # real cval to out-of-bounds pixels ourselves.
            result = self._ndimage.map_coordinates(
                input, src, order=order, mode="constant", cval=0.0,
            )
            oob = jnp.zeros(output_shape, dtype=bool)
            for d in range(ndim):
                oob = oob | (src[d] < 0) | (src[d] > input.shape[d] - 1)
            result = jnp.where(oob, cval, result)
        else:
            result = self._ndimage.map_coordinates(
                input, src, order=order, mode=mode, cval=cval,
            )

        return result

    def __repr__(self):
        return "JaxBackend()"
