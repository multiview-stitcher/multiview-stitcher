"""Legacy NumPy/SciPy backend -- kept for benchmark comparison only.

.. deprecated::
    Use ``get_backend("numpy")`` instead, which returns an
    ``ArrayAPIBackend("numpy")`` backed by the shared ``XPBackend`` base.
"""

import warnings

import numpy as np
from scipy.ndimage import (
    affine_transform,
    distance_transform_edt,
    gaussian_filter,
)
from skimage.exposure import rescale_intensity
from skimage.metrics import structural_similarity
from skimage.registration import phase_cross_correlation

from multiview_stitcher.backends._base import Backend


class NumpyLegacyBackend(Backend):
    """Legacy NumPy backend -- deprecated, kept for A/B benchmarking."""

    def __init__(self):
        warnings.warn(
            "NumpyLegacyBackend is deprecated and kept only for benchmark "
            "comparison. Use get_backend('numpy') instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def asarray(self, x, dtype=None):
        return np.asarray(x, dtype=dtype)

    def to_numpy(self, x):
        return np.asarray(x)

    def zeros(self, shape, dtype=None):
        return np.zeros(shape, dtype=dtype)

    def ones_like(self, x):
        return np.ones_like(x)

    def stack(self, arrays, axis=0):
        return np.stack(arrays, axis=axis)

    def array(self, x):
        return np.array(x)

    def nansum(self, x, axis=None):
        return np.nansum(x, axis=axis)

    def nanmax(self, x, axis=None):
        return np.nanmax(x, axis=axis)

    def nanmin(self, x, axis=None):
        return np.nanmin(x, axis=axis)

    def nan_to_num(self, x, nan=0.0):
        return np.nan_to_num(x, nan=nan)

    def isnan(self, x):
        return np.isnan(x)

    def any(self, x):
        return np.any(x)

    def sum(self, x, axis=None):
        return np.sum(x, axis=axis)

    def abs(self, x):
        return np.abs(x)

    def cos(self, x):
        return np.cos(x)

    def clip(self, x, a_min, a_max):
        return np.clip(x, a_min, a_max)

    def where(self, condition, x, y):
        return np.where(condition, x, y)

    @property
    def pi(self):
        return np.pi

    @property
    def nan(self):
        return np.nan

    @property
    def newaxis(self):
        return np.newaxis

    def affine_transform(
        self, input, matrix, offset, output_shape,
        mode="constant", cval=0.0, order=1,
    ):
        return affine_transform(
            input, matrix=matrix, offset=offset,
            output_shape=output_shape, mode=mode, cval=cval, order=order,
        )

    def distance_transform_edt(self, input, sampling=None):
        return distance_transform_edt(input, sampling=sampling)

    def gaussian_filter(self, input, sigma, **kwargs):
        return gaussian_filter(input, sigma, **kwargs)

    def rescale_intensity(self, image, in_range, out_range):
        return rescale_intensity(image, in_range=in_range, out_range=out_range)

    def phase_cross_correlation(
        self, reference_image, moving_image, **kwargs,
    ):
        return phase_cross_correlation(
            reference_image, moving_image, **kwargs,
        )

    def structural_similarity(self, im1, im2, **kwargs):
        return structural_similarity(im1, im2, **kwargs)

    def masked_fill(self, array, mask, value):
        return np.where(mask, value, array)

    def normalize_weights(self, weights):
        wsum = np.nansum(weights, axis=0)
        wsum = np.where(wsum == 0, np.ones_like(wsum), wsum)
        return weights / wsum

    def fused_weighted_nansum(self, images, weights):
        return np.nansum(images * weights, axis=0)

    @property
    def recommended_dask_scheduler(self):
        return None
