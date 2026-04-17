"""Unified Array API backend for numpy, cupy, and dpnp.

This backend uses a single code path for multiple array libraries by
dispatching operations through the array module's namespace (``xp``).
For scipy.ndimage and skimage operations, it dispatches to the
appropriate GPU-accelerated library when available, falling back to
CPU scipy/skimage otherwise.

Supported modules
-----------------
``"numpy"`` -- CPU, delegates to numpy + scipy + skimage
``"cupy"``  -- NVIDIA GPU, delegates to cupy + cupyx.scipy + cucim
``"dpnp"``  -- Intel XPU, delegates to dpnp + CPU scipy/skimage fallback
"""

import importlib
import logging
import os

import numpy as np

from multiview_stitcher.backends._xp_backend import XPBackend

logger = logging.getLogger(__name__)

# Backends whose arrays live on a device (GPU / XPU).
_GPU_MODULES = {"cupy", "dpnp"}

# Per-device cache: device_id -> bool (True = rocBLAS lacks Tensile kernels).
_rocblas_unsupported_cache: dict[int, bool] = {}


def _rocblas_missing_for_device(device_id: int) -> bool:
    """Check whether rocBLAS has Tensile kernel libraries for a HIP device.

    rocBLAS abort()s (SIGABRT, uncatchable) when asked to run on a GPU
    whose architecture has no ``TensileLibrary_lazy_gfx<ARCH>.dat`` file.
    This function probes the filesystem so we can route to a CPU fallback
    before that happens.
    """
    if device_id in _rocblas_unsupported_cache:
        return _rocblas_unsupported_cache[device_id]

    try:
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(device_id)
        arch_raw = props.get("gcnArchName", b"")
        arch_raw = arch_raw.decode() if isinstance(arch_raw, bytes) else str(arch_raw)
        # e.g. "gfx906:sramecc+:xnack-" -> "gfx906"
        arch = arch_raw.split(":")[0]

        rocm_home = os.environ.get("ROCM_HOME", "/opt/rocm")
        tensile_dir = os.path.join(rocm_home, "lib", "rocblas", "library")
        tensile_file = os.path.join(tensile_dir, f"TensileLibrary_lazy_{arch}.dat")

        missing = not os.path.isfile(tensile_file)
        if missing:
            device_name = props.get("name", b"")
            if isinstance(device_name, bytes):
                device_name = device_name.decode()
            logger.warning(
                "rocBLAS has no Tensile kernels for %s (%s) — "
                "hipCIM operations (phase_cross_correlation, "
                "structural_similarity) will fall back to CPU skimage. "
                "Looked for: %s",
                device_name, arch, tensile_file,
            )
    except Exception:
        missing = False

    _rocblas_unsupported_cache[device_id] = missing
    return missing


def _hipcim_current_device_unsupported() -> bool:
    """True if hipCIM calls would crash on the active CuPy device.

    On NVIDIA (CUDA) this always returns False — cucim failures there are
    catchable Python exceptions.  On AMD (HIP) we probe rocBLAS's Tensile
    library directory to avoid an uncatchable SIGABRT.
    """
    try:
        import cupy as cp

        if not cp.cuda.runtime.is_hip:
            return False
        return _rocblas_missing_for_device(cp.cuda.Device().id)
    except Exception:
        return False


class ArrayAPIBackend(XPBackend):
    """Backend that delegates all operations to a given array module.

    Parameters
    ----------
    module_name : str
        Name of the array module (``"numpy"``, ``"cupy"``, or ``"dpnp"``).
    """

    def __init__(self, module_name):
        xp = importlib.import_module(module_name)
        super().__init__(xp, name=module_name)
        self._is_gpu = module_name in _GPU_MODULES

        # Capability flags — cupy has native GPU implementations
        if module_name == "cupy":
            self._has_native_affine_transform = True
            self._has_native_phase_cross_correlation = True
            self._has_native_structural_similarity = True

        # numpy and numba don't need fallback warnings (they ARE the fallback)
        if module_name == "numpy":
            self._suppress_fallback_warnings = {
                "affine_transform",
                "gaussian_filter",
                "phase_cross_correlation",
                "structural_similarity",
            }

    # -- Array conversion (module-specific) ---------------------------------

    def to_numpy(self, x):
        if self._name == "numpy":
            return np.asarray(x)
        if self._name == "cupy":
            if hasattr(x, "get"):
                return x.get()
            return np.asarray(x)
        if self._name == "dpnp":
            dpnp = self.xp
            if isinstance(x, dpnp.ndarray):
                return dpnp.asnumpy(x)
            return np.asarray(x)
        # Generic fallback
        if hasattr(x, "get"):
            return x.get()
        if hasattr(x, "asnumpy"):
            return x.asnumpy()
        return np.asarray(x)

    # -- Native GPU implementations (cupy only) -----------------------------

    def _native_affine_transform(
        self, input, matrix, offset, output_shape,
        mode="constant", cval=0.0, order=1,
    ):
        import cupy as cp
        import cupyx.scipy.ndimage as cpx_ndimage

        return cpx_ndimage.affine_transform(
            input,
            matrix=cp.asarray(matrix),
            offset=cp.asarray(offset),
            output_shape=output_shape,
            mode=mode, cval=cval, order=order,
        )

    def _native_phase_cross_correlation(
        self, reference_image, moving_image, **kwargs,
    ):
        if _hipcim_current_device_unsupported():
            return self._skimage_phase_cross_correlation(
                reference_image, moving_image, **kwargs,
            )
        try:
            from cucim.skimage.registration import (
                phase_cross_correlation as _gpu_pcc,
            )
            return _gpu_pcc(reference_image, moving_image, **kwargs)
        except ImportError:
            # cucim not available — fall back to CPU
            return self._skimage_phase_cross_correlation(
                reference_image, moving_image, **kwargs,
            )

    def _native_structural_similarity(self, im1, im2, **kwargs):
        if _hipcim_current_device_unsupported():
            return self._skimage_structural_similarity(im1, im2, **kwargs)
        try:
            from cucim.skimage.metrics import (
                structural_similarity as _gpu_ssim,
            )
            return _gpu_ssim(im1, im2, **kwargs)
        except ImportError:
            return self._skimage_structural_similarity(im1, im2, **kwargs)

    # -- Fusion helpers (numpy: use numba, GPU: use array-API) ---------------

    def normalize_weights(self, weights):
        if self._name == "numpy":
            from multiview_stitcher._numba_acceleration import (
                normalize_weights as _accel_normalize,
            )
            return _accel_normalize(weights)
        return super().normalize_weights(weights)

    def fused_weighted_nansum(self, images, weights):
        if self._name == "numpy":
            from multiview_stitcher._numba_acceleration import (
                fused_weighted_nansum as _accel_fwns,
            )
            return _accel_fwns(images, weights)
        return super().fused_weighted_nansum(images, weights)

    # -- Memory management (cupy only) --------------------------------------

    def free_memory(self):
        if self._name == "cupy":
            try:
                self.xp.cuda.Device().synchronize()
                self.xp.get_default_memory_pool().free_all_blocks()
                self.xp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass

    def __repr__(self):
        return f"ArrayAPIBackend({self._name!r})"
