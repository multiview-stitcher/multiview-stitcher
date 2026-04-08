"""Legacy CuPy/cupyx backend -- kept for benchmark comparison only.

.. deprecated::
    Use ``get_backend("cupy")`` instead, which returns an
    ``ArrayAPIBackend("cupy")`` backed by the shared ``XPBackend`` base.
"""

import warnings

import cupy as cp
import cupyx.scipy.ndimage as cpx_ndimage
import numpy as np
from cupy.cuda import runtime as _cuda_runtime
from scipy.ndimage import distance_transform_edt as _cpu_distance_transform_edt
from scipy.ndimage import gaussian_filter as _cpu_gaussian_filter

from multiview_stitcher.backends._base import Backend

_IS_HIP = _cuda_runtime.is_hip

# Optional: cucim provides GPU-native skimage functions
try:
    from cucim.skimage.registration import phase_cross_correlation as _gpu_phase_cross_correlation
    _HAS_CUCIM_REGISTRATION = True
except ImportError:
    _HAS_CUCIM_REGISTRATION = False

try:
    from cucim.skimage.metrics import structural_similarity as _gpu_structural_similarity
    _HAS_CUCIM_METRICS = True
except ImportError:
    _HAS_CUCIM_METRICS = False


class CupyLegacyBackend(Backend):
    """Legacy CuPy backend -- deprecated, kept for A/B benchmarking."""

    def __init__(self):
        warnings.warn(
            "CupyLegacyBackend is deprecated and kept only for benchmark "
            "comparison. Use get_backend('cupy') instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def asarray(self, x, dtype=None):
        arr = cp.asarray(x)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    def to_numpy(self, x):
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
        return np.asarray(x)

    def zeros(self, shape, dtype=None):
        return cp.zeros(shape, dtype=dtype)

    def ones_like(self, x):
        return cp.ones_like(x)

    def stack(self, arrays, axis=0):
        return cp.stack(arrays, axis=axis)

    def array(self, x):
        return cp.array(x)

    def nansum(self, x, axis=None):
        return cp.nansum(x, axis=axis)

    def nanmax(self, x, axis=None):
        return cp.nanmax(x, axis=axis)

    def nanmin(self, x, axis=None):
        return cp.nanmin(x, axis=axis)

    def nan_to_num(self, x, nan=0.0):
        return cp.nan_to_num(x, nan=nan)

    def isnan(self, x):
        return cp.isnan(x)

    def any(self, x):
        return cp.any(x)

    def sum(self, x, axis=None):
        return cp.sum(x, axis=axis)

    def abs(self, x):
        return cp.abs(x)

    def cos(self, x):
        return cp.cos(x)

    def clip(self, x, a_min, a_max):
        return cp.clip(x, a_min, a_max)

    @property
    def pi(self):
        return cp.pi

    @property
    def nan(self):
        return cp.nan

    @property
    def newaxis(self):
        return cp.newaxis

    def affine_transform(
        self, input, matrix, offset, output_shape,
        mode="constant", cval=0.0, order=1,
    ):
        return cpx_ndimage.affine_transform(
            input,
            matrix=cp.asarray(matrix),
            offset=cp.asarray(offset),
            output_shape=output_shape,
            mode=mode,
            cval=cval,
            order=order,
        )

    def distance_transform_edt(self, input, sampling=None):
        cpu_input = cp.asnumpy(input) if isinstance(input, cp.ndarray) else np.asarray(input)
        result = _cpu_distance_transform_edt(cpu_input, sampling=sampling)
        return cp.asarray(result)

    def gaussian_filter(self, input, sigma, **kwargs):
        cpu_input = cp.asnumpy(input) if isinstance(input, cp.ndarray) else np.asarray(input)
        result = _cpu_gaussian_filter(cpu_input, sigma, **kwargs)
        return cp.asarray(result)

    def rescale_intensity(self, image, in_range, out_range):
        imin, imax = in_range
        omin, omax = out_range
        scale = (omax - omin) / (imax - imin) if imax != imin else 0
        return cp.clip((image - imin) * scale + omin, omin, omax)

    def phase_cross_correlation(
        self, reference_image, moving_image, **kwargs,
    ):
        if _HAS_CUCIM_REGISTRATION:
            return _gpu_phase_cross_correlation(
                reference_image, moving_image, **kwargs,
            )
        from skimage.registration import phase_cross_correlation as _cpu_pcc
        ref_np = cp.asnumpy(reference_image) if isinstance(reference_image, cp.ndarray) else np.asarray(reference_image)
        mov_np = cp.asnumpy(moving_image) if isinstance(moving_image, cp.ndarray) else np.asarray(moving_image)
        for key in ("reference_mask", "moving_mask"):
            if key in kwargs and isinstance(kwargs[key], cp.ndarray):
                kwargs[key] = cp.asnumpy(kwargs[key])
        return _cpu_pcc(ref_np, mov_np, **kwargs)

    def structural_similarity(self, im1, im2, **kwargs):
        if _HAS_CUCIM_METRICS:
            return _gpu_structural_similarity(im1, im2, **kwargs)
        from skimage.metrics import structural_similarity as _cpu_ssim
        im1_np = cp.asnumpy(im1) if isinstance(im1, cp.ndarray) else np.asarray(im1)
        im2_np = cp.asnumpy(im2) if isinstance(im2, cp.ndarray) else np.asarray(im2)
        return _cpu_ssim(im1_np, im2_np, **kwargs)

    def masked_fill(self, array, mask, value):
        if _IS_HIP:
            return cp.where(mask, value, array)
        array[mask] = value
        return array

    def normalize_weights(self, weights):
        """Normalize weights along axis 0."""
        wsum = cp.nansum(weights, axis=0)
        wsum = cp.where(wsum == 0, cp.ones_like(wsum), wsum)
        return weights / wsum

    def fused_weighted_nansum(self, images, weights):
        """Compute nansum(images * weights, axis=0) in one pass."""
        return cp.nansum(images * weights, axis=0)

    def free_memory(self):
        try:
            cp.cuda.Device().synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    @property
    def recommended_dask_scheduler(self):
        return "synchronous"
