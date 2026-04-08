"""MLX backend for GPU-accelerated computation on Apple Silicon.

Uses Apple's MLX framework to run array operations on M-series GPU.
Key characteristics:

* **Unified memory** -- no CPU<->GPU data copies on Apple Silicon.
  Fallback to CPU scipy for ndimage operations is essentially free.
* **float32 only** -- MLX does not support float64.  Input data is
  automatically converted to float32.
* **nan-reductions** -- ``nansum``, ``nanmax``, ``nanmin`` are
  implemented via ``mx.where(mx.isnan(...), ...)`` since MLX does not
  provide them natively.
"""

import warnings

import numpy as np

from multiview_stitcher.backends._xp_backend import XPBackend

# Lazy import -- only loaded when the backend is actually used.
_mx = None


def _ensure_mlx():
    global _mx
    if _mx is None:
        import mlx.core as mx
        _mx = mx
    return _mx


class MLXBackend(XPBackend):
    """Apple Silicon GPU backend via MLX.

    Falls back to CPU scipy / skimage for operations that MLX does
    not provide (ndimage, skimage metrics).  Thanks to unified memory
    these round-trips are nearly free.
    """

    _is_gpu = True

    # Suppress fallback warnings -- unified memory means CPU fallbacks
    # are essentially free on Apple Silicon.
    _suppress_fallback_warnings = {
        "affine_transform",
        "gaussian_filter",
        "phase_cross_correlation",
        "structural_similarity",
    }

    def __init__(self):
        mx = _ensure_mlx()
        super().__init__(mx, name="mlx")

    # -- Helpers ------------------------------------------------------------

    def _f32(self, x):
        """Ensure array is float32 (MLX does not support float64)."""
        if hasattr(x, "dtype") and x.dtype == self.xp.float64:
            self._warn_precision_loss(np.float64, np.float32)
            return x.astype(self.xp.float32)
        return x

    def _warn_precision_loss(self, from_dtype, to_dtype):
        warnings.warn(
            f"MLXBackend: casting {np.dtype(from_dtype)} to "
            f"{np.dtype(to_dtype)} — MLX does not support float64. "
            f"This may reduce numerical precision.",
            stacklevel=3,
        )

    # -- Array creation / conversion (float32 coercion) ---------------------

    def asarray(self, x, dtype=None):
        arr = self.xp.array(np.asarray(x))
        arr = self._f32(arr)
        if dtype is not None and np.dtype(dtype) == np.float64:
            self._warn_precision_loss(dtype, np.float32)
            dtype = np.float32
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    def to_numpy(self, x):
        return np.array(x)

    def zeros(self, shape, dtype=None):
        if dtype is not None and np.dtype(dtype) == np.float64:
            self._warn_precision_loss(dtype, np.float32)
            dtype = np.float32
        return self.xp.zeros(shape, dtype=dtype)

    def array(self, x):
        return self._f32(self.xp.array(np.asarray(x)))

    # -- Math operations (MLX lacks nan-aware reductions) -------------------

    def nansum(self, x, axis=None):
        mx = self.xp
        safe = mx.where(mx.isnan(x), 0, x)
        return mx.sum(safe, axis=axis)

    def nanmax(self, x, axis=None):
        mx = self.xp
        neg_inf = mx.array(float("-inf"), dtype=x.dtype)
        safe = mx.where(mx.isnan(x), neg_inf, x)
        return mx.max(safe, axis=axis)

    def nanmin(self, x, axis=None):
        mx = self.xp
        pos_inf = mx.array(float("inf"), dtype=x.dtype)
        safe = mx.where(mx.isnan(x), pos_inf, x)
        return mx.min(safe, axis=axis)

    def nan_to_num(self, x, nan=0.0):
        mx = self.xp
        return mx.where(mx.isnan(x), nan, x)

    # -- Properties (MLX-specific) ------------------------------------------

    @property
    def pi(self):
        return float(np.pi)

    @property
    def nan(self):
        return float("nan")

    @property
    def newaxis(self):
        return None  # same as np.newaxis

    def masked_fill(self, array, mask, value):
        return self.xp.where(mask, value, array)

    def __repr__(self):
        return "MLXBackend()"
