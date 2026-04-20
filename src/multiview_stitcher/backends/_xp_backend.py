"""Shared base class for array-module backends.

``XPBackend`` implements the full ``Backend`` interface by delegating
array operations to an *array module* (``xp``) — e.g. ``numpy``,
``cupy``, ``jax.numpy``, or ``mlx.core``.  Subclasses only need to
override the methods where their behaviour differs from the default.

Automatic delegation
--------------------
Simple operations that map 1:1 to ``xp.<name>`` are delegated
automatically.  To add a new pass-through, just add its name to
``_XP_DELEGATES`` — no method body needed.  This works for both
callable functions and value attributes (``pi``, ``nan``, ``newaxis``).

Capability flags
----------------
Subclasses declare which operations they implement natively (on-device)
by setting class attributes:

    _has_native_affine_transform = True
    _has_native_phase_cross_correlation = True
    ...

When a flag is ``False`` (the default), the base class falls back to a
CPU scipy/skimage implementation and emits a one-shot performance
warning (suppressible via ``_suppress_fallback_warnings``).
"""

import warnings

import numpy as np

from multiview_stitcher.backends._base import Backend

# ---------------------------------------------------------------------------
# Automatic xp delegation
# ---------------------------------------------------------------------------
# Names listed here are resolved to ``self.xp.<name>`` at access time.
# To add a new pass-through operation, just add its name to the
# appropriate set — no method body required.

_XP_DELEGATES = {
    # array creation / conversion
    "asarray", "zeros", "ones_like", "stack", "array",
    # math
    "nansum", "nanmax", "nanmin", "nan_to_num", "isnan",
    "any", "sum", "abs", "cos", "sin", "clip",
    # selection
    "where",
    # constants / values
    "pi", "nan", "newaxis",
}

# ---------------------------------------------------------------------------
# GPU module names — used to set dask scheduler hint
# ---------------------------------------------------------------------------
_GPU_MODULES = {"cupy", "dpnp"}


class XPBackend(Backend):
    """Base backend that delegates operations to an array module ``xp``.

    Simple operations (those in ``_XP_DELEGATES``) are forwarded to
    ``self.xp`` automatically.  Subclasses can still override any of
    them by defining a regular method or attribute.

    Parameters
    ----------
    xp : module
        The array module (e.g. ``numpy``, ``cupy``, ``jax.numpy``).
    name : str, optional
        Human-readable backend name for warnings/repr.  Defaults to
        ``xp.__name__``.
    """

    # -- Capability flags (subclasses override) -----------------------------
    _has_native_affine_transform = False
    _has_native_phase_cross_correlation = False
    _has_native_structural_similarity = False
    _has_native_gaussian_filter = False
    _is_gpu = False

    # Set of operation names for which fallback warnings are suppressed.
    # Subclasses can override, e.g. MLX where scipy fallback is ~free.
    _suppress_fallback_warnings: set[str] = set()

    def __init__(self, xp, *, name=None):
        self.xp = xp
        self._name = name or xp.__name__
        self._warned: set[str] = set()
        # Bind xp functions as instance attributes so they shadow the
        # NotImplementedError stubs in Backend.  Subclass __init__ can
        # overwrite any of these after calling super().__init__().
        for attr in _XP_DELEGATES:
            if attr not in type(self).__dict__:
                # Only bind if the subclass hasn't explicitly overridden it
                setattr(self, attr, getattr(xp, attr))

    # -- Fallback warning helper --------------------------------------------

    def _warn_fallback(self, operation: str):
        """Emit a one-shot warning that *operation* is using a CPU fallback."""
        if operation in self._warned:
            return
        if operation in self._suppress_fallback_warnings:
            self._warned.add(operation)
            return
        self._warned.add(operation)
        warnings.warn(
            f"{self!r}: {operation} is falling back to CPU "
            f"(no native implementation). This may be slow for "
            f"large arrays on GPU/accelerator backends.",
            stacklevel=3,
        )

    # -- Explicitly defined methods -----------------------------------------
    # Only operations with non-trivial logic need a method body.

    def to_numpy(self, x):
        return np.asarray(x)

    def masked_fill(self, array, mask, value):
        return self.xp.where(mask, value, array)

    # -- ndimage operations (capability-flag dispatch) ----------------------

    def affine_transform(
        self, input, matrix, offset, output_shape,
        mode="constant", cval=0.0, order=1,
    ):
        if self._has_native_affine_transform:
            return self._native_affine_transform(
                input, matrix, offset, output_shape,
                mode=mode, cval=cval, order=order,
            )
        self._warn_fallback("affine_transform")
        return self._scipy_affine_transform(
            input, matrix, offset, output_shape,
            mode=mode, cval=cval, order=order,
        )

    def _native_affine_transform(
        self, input, matrix, offset, output_shape,
        mode="constant", cval=0.0, order=1,
    ):
        raise NotImplementedError(
            f"{self!r} declares _has_native_affine_transform but "
            f"does not implement _native_affine_transform"
        )

    def _scipy_affine_transform(
        self, input, matrix, offset, output_shape,
        mode="constant", cval=0.0, order=1,
    ):
        from multiview_stitcher._numba_acceleration import (
            affine_transform as _accel_affine,
        )

        cpu_input = self.to_numpy(input)
        result = _accel_affine(
            cpu_input,
            matrix=np.asarray(matrix),
            offset=np.asarray(offset),
            output_shape=output_shape,
            mode=mode, cval=cval, order=order,
        )
        return self.asarray(result)

    def distance_transform_edt(self, input, sampling=None):
        from scipy.ndimage import distance_transform_edt

        cpu_input = self.to_numpy(input)
        result = distance_transform_edt(cpu_input, sampling=sampling)
        return self.asarray(result)

    def gaussian_filter(self, input, sigma, **kwargs):
        if self._has_native_gaussian_filter:
            return self._native_gaussian_filter(input, sigma, **kwargs)
        self._warn_fallback("gaussian_filter")
        return self._scipy_gaussian_filter(input, sigma, **kwargs)

    def _native_gaussian_filter(self, input, sigma, **kwargs):
        raise NotImplementedError

    def _scipy_gaussian_filter(self, input, sigma, **kwargs):
        from scipy.ndimage import gaussian_filter

        cpu_input = self.to_numpy(input)
        result = gaussian_filter(cpu_input, sigma, **kwargs)
        return self.asarray(result)

    # -- Image processing (capability-flag dispatch) ------------------------

    def rescale_intensity(self, image, in_range, out_range):
        imin, imax = in_range
        omin, omax = out_range
        scale = (omax - omin) / (imax - imin) if imax != imin else 0
        return self.xp.clip((image - imin) * scale + omin, omin, omax)

    def phase_cross_correlation(
        self, reference_image, moving_image, **kwargs,
    ):
        if self._has_native_phase_cross_correlation:
            return self._native_phase_cross_correlation(
                reference_image, moving_image, **kwargs,
            )
        self._warn_fallback("phase_cross_correlation")
        return self._skimage_phase_cross_correlation(
            reference_image, moving_image, **kwargs,
        )

    def _native_phase_cross_correlation(
        self, reference_image, moving_image, **kwargs,
    ):
        raise NotImplementedError

    def _skimage_phase_cross_correlation(
        self, reference_image, moving_image, **kwargs,
    ):
        from skimage.registration import phase_cross_correlation

        ref_np = self.to_numpy(reference_image)
        mov_np = self.to_numpy(moving_image)
        for key in ("reference_mask", "moving_mask"):
            if key in kwargs:
                kwargs[key] = self.to_numpy(kwargs[key])
        return phase_cross_correlation(ref_np, mov_np, **kwargs)

    def structural_similarity(self, im1, im2, **kwargs):
        if self._has_native_structural_similarity:
            return self._native_structural_similarity(im1, im2, **kwargs)
        self._warn_fallback("structural_similarity")
        return self._skimage_structural_similarity(im1, im2, **kwargs)

    def _native_structural_similarity(self, im1, im2, **kwargs):
        raise NotImplementedError

    def _skimage_structural_similarity(self, im1, im2, **kwargs):
        from skimage.metrics import structural_similarity

        return structural_similarity(
            self.to_numpy(im1), self.to_numpy(im2), **kwargs,
        )

    # -- Fusion helpers (pure array-API, stays on-device) -------------------

    def normalize_weights(self, weights):
        """Normalize weights along axis 0."""
        wsum = self.nansum(weights, axis=0)
        wsum = self.xp.where(wsum == 0, self.xp.ones_like(wsum), wsum)
        return weights / wsum

    def fused_weighted_nansum(self, images, weights):
        """Compute nansum(images * weights, axis=0) in one pass."""
        return self.nansum(images * weights, axis=0)

    # -- Dask integration ---------------------------------------------------

    @property
    def recommended_dask_scheduler(self):
        return "synchronous" if self._is_gpu else None

    # -- Memory management --------------------------------------------------

    def free_memory(self):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self._name!r})"
