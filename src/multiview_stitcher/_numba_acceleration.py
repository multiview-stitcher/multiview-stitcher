"""Transparent numba acceleration for CPU code paths.

When numba is installed and acceleration is enabled (the default), hot-path
operations automatically use JIT-compiled parallel kernels.  When numba is
not installed or acceleration is disabled, the functions fall back to
plain numpy/scipy — so callers never need to check.

Toggle from user code::

    import multiview_stitcher as msv
    msv.set_numba_acceleration(False)   # disable for benchmarking
    msv.set_numba_acceleration(True)    # re-enable

The kernels are compiled lazily on first use and cached to disk, so the
~1 s numba import cost is only paid when a kernel is actually called.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Availability detection and toggle
# ---------------------------------------------------------------------------

_numba_available = False
try:
    import numba as _nb  # noqa: F401

    _numba_available = True
    del _nb
except ImportError:
    pass

_use_numba_acceleration = _numba_available


def set_numba_acceleration(enabled: bool):
    """Enable or disable numba acceleration package-wide.

    When enabled (the default if numba is installed), compute-intensive
    inner loops use JIT-compiled kernels for significant speedups.

    Disabling is useful for benchmarking pure-numpy/backend performance
    or for debugging.

    Parameters
    ----------
    enabled : bool
        Whether to use numba acceleration.

    Raises
    ------
    ImportError
        If ``enabled=True`` but numba is not installed.
    """
    global _use_numba_acceleration
    if enabled and not _numba_available:
        raise ImportError(
            "Cannot enable numba acceleration: numba is not installed. "
            "Install it with: pip install numba"
        )
    _use_numba_acceleration = enabled


def get_numba_acceleration() -> bool:
    """Return whether numba acceleration is currently enabled."""
    return _use_numba_acceleration


def numba_available() -> bool:
    """Return whether numba is installed and importable."""
    return _numba_available


# ---------------------------------------------------------------------------
# Lazy kernel compilation
# ---------------------------------------------------------------------------

_KERNELS_READY = False

# Module-level names filled by _compile_kernels
_affine_transform_2d = None
_affine_transform_3d = None
_cosine_weights_nb = None
_normalize_weights_3d_nb = None
_normalize_weights_2d_nb = None
_fused_weighted_nansum_3d = None
_fused_weighted_nansum_2d = None
_nan_to_num_nb = None


def _ensure_kernels():
    """Compile all numba kernels (once, cached to disk)."""
    global _KERNELS_READY
    if _KERNELS_READY:
        return
    _compile_kernels()
    _KERNELS_READY = True


def _compile_kernels():
    global _KERNELS_READY
    global _affine_transform_2d, _affine_transform_3d
    global _cosine_weights_nb
    global _normalize_weights_3d_nb, _normalize_weights_2d_nb
    global _fused_weighted_nansum_3d, _fused_weighted_nansum_2d
    global _nan_to_num_nb

    from numba import njit, prange

    # ------------------------------------------------------------------
    # 3-D affine transform (trilinear interpolation, parallelised over z)
    # ------------------------------------------------------------------
    @njit(parallel=True, cache=True)
    def __affine_transform_3d(source, matrix, offset, out_z, out_y, out_x,
                              cval, order):
        iz, iy, ix = source.shape
        output = np.empty((out_z, out_y, out_x), dtype=np.float64)

        for z in prange(out_z):
            for y in range(out_y):
                for x in range(out_x):
                    sz = (matrix[0, 0] * z + matrix[0, 1] * y
                          + matrix[0, 2] * x + offset[0])
                    sy = (matrix[1, 0] * z + matrix[1, 1] * y
                          + matrix[1, 2] * x + offset[1])
                    sx = (matrix[2, 0] * z + matrix[2, 1] * y
                          + matrix[2, 2] * x + offset[2])

                    if order == 0:
                        rz = int(round(sz))
                        ry = int(round(sy))
                        rx = int(round(sx))
                        if 0 <= rz < iz and 0 <= ry < iy and 0 <= rx < ix:
                            output[z, y, x] = source[rz, ry, rx]
                        else:
                            output[z, y, x] = cval
                    else:
                        if (sz < 0.0 or sz > iz - 1.0
                                or sy < 0.0 or sy > iy - 1.0
                                or sx < 0.0 or sx > ix - 1.0):
                            output[z, y, x] = cval
                            continue

                        z0 = int(np.floor(sz))
                        y0 = int(np.floor(sy))
                        x0 = int(np.floor(sx))
                        dz = sz - z0
                        dy = sy - y0
                        dx = sx - x0

                        val = 0.0
                        for kz in range(2):
                            zz = min(z0 + kz, iz - 1)
                            wz = (1.0 - dz) if kz == 0 else dz
                            for ky in range(2):
                                yy = min(y0 + ky, iy - 1)
                                wy = (1.0 - dy) if ky == 0 else dy
                                for kx in range(2):
                                    xx = min(x0 + kx, ix - 1)
                                    wx = (1.0 - dx) if kx == 0 else dx
                                    val += wz * wy * wx * source[zz, yy, xx]
                        output[z, y, x] = val
        return output

    # ------------------------------------------------------------------
    # 2-D affine transform (bilinear interpolation, parallelised over y)
    # ------------------------------------------------------------------
    @njit(parallel=True, cache=True)
    def __affine_transform_2d(source, matrix, offset, out_y, out_x,
                              cval, order):
        iy, ix = source.shape
        output = np.empty((out_y, out_x), dtype=np.float64)

        for y in prange(out_y):
            for x in range(out_x):
                sy = matrix[0, 0] * y + matrix[0, 1] * x + offset[0]
                sx = matrix[1, 0] * y + matrix[1, 1] * x + offset[1]

                if order == 0:
                    ry = int(round(sy))
                    rx = int(round(sx))
                    if 0 <= ry < iy and 0 <= rx < ix:
                        output[y, x] = source[ry, rx]
                    else:
                        output[y, x] = cval
                else:
                    if (sy < 0.0 or sy > iy - 1.0
                            or sx < 0.0 or sx > ix - 1.0):
                        output[y, x] = cval
                        continue

                    y0 = int(np.floor(sy))
                    x0 = int(np.floor(sx))
                    dy = sy - y0
                    dx = sx - x0

                    val = 0.0
                    for ky in range(2):
                        yy = min(y0 + ky, iy - 1)
                        wy = (1.0 - dy) if ky == 0 else dy
                        for kx in range(2):
                            xx = min(x0 + kx, ix - 1)
                            wx = (1.0 - dx) if kx == 0 else dx
                            val += wy * wx * source[yy, xx]
                    output[y, x] = val
        return output

    # ------------------------------------------------------------------
    # Cosine weighting (works for any dimensionality via ravel)
    # ------------------------------------------------------------------
    @njit(parallel=True, cache=True)
    def __cosine_weights_nb(x):
        flat = x.ravel()
        n = flat.shape[0]
        for i in prange(n):
            v = flat[i]
            if v < 1.0:
                flat[i] = (np.cos((1.0 - v) * np.pi) + 1.0) / 2.0
            if flat[i] < 0.0:
                flat[i] = 0.0
            elif flat[i] > 1.0:
                flat[i] = 1.0
        return x

    # ------------------------------------------------------------------
    # Weight normalisation (3-D and 2-D)
    # ------------------------------------------------------------------
    @njit(parallel=True, cache=True)
    def __normalize_weights_3d_nb(w):
        n_views, nz, ny, nx = w.shape
        for z in prange(nz):
            for y in range(ny):
                for x in range(nx):
                    s = 0.0
                    for v in range(n_views):
                        val = w[v, z, y, x]
                        if not np.isnan(val):
                            s += val
                    if s == 0.0:
                        s = 1.0
                    for v in range(n_views):
                        w[v, z, y, x] /= s
        return w

    @njit(parallel=True, cache=True)
    def __normalize_weights_2d_nb(w):
        n_views, ny, nx = w.shape
        for y in prange(ny):
            for x in range(nx):
                s = 0.0
                for v in range(n_views):
                    val = w[v, y, x]
                    if not np.isnan(val):
                        s += val
                if s == 0.0:
                    s = 1.0
                for v in range(n_views):
                    w[v, y, x] /= s
        return w

    # ------------------------------------------------------------------
    # Fused weighted nansum (3-D and 2-D)
    # ------------------------------------------------------------------
    @njit(parallel=True, cache=True)
    def __fused_weighted_nansum_3d(images, weights):
        n_views, nz, ny, nx = images.shape
        result = np.empty((nz, ny, nx), dtype=np.float64)
        for z in prange(nz):
            for y in range(ny):
                for x in range(nx):
                    s = 0.0
                    for v in range(n_views):
                        val = images[v, z, y, x] * weights[v, z, y, x]
                        if not np.isnan(val):
                            s += val
                    result[z, y, x] = s
        return result

    @njit(parallel=True, cache=True)
    def __fused_weighted_nansum_2d(images, weights):
        n_views, ny, nx = images.shape
        result = np.empty((ny, nx), dtype=np.float64)
        for y in prange(ny):
            for x in range(nx):
                s = 0.0
                for v in range(n_views):
                    val = images[v, y, x] * weights[v, y, x]
                    if not np.isnan(val):
                        s += val
                result[y, x] = s
        return result

    # ------------------------------------------------------------------
    # nan_to_num (flat, any dimensionality)
    # ------------------------------------------------------------------
    @njit(parallel=True, cache=True)
    def __nan_to_num_nb(arr):
        flat = arr.ravel()
        n = flat.shape[0]
        for i in prange(n):
            if np.isnan(flat[i]):
                flat[i] = 0.0
        return arr

    # Assign to module-level names
    _affine_transform_2d = __affine_transform_2d
    _affine_transform_3d = __affine_transform_3d
    _cosine_weights_nb = __cosine_weights_nb
    _normalize_weights_3d_nb = __normalize_weights_3d_nb
    _normalize_weights_2d_nb = __normalize_weights_2d_nb
    _fused_weighted_nansum_3d = __fused_weighted_nansum_3d
    _fused_weighted_nansum_2d = __fused_weighted_nansum_2d
    _nan_to_num_nb = __nan_to_num_nb

    _KERNELS_READY = True


# ---------------------------------------------------------------------------
# Public wrapper functions
# ---------------------------------------------------------------------------
# Each function checks the toggle, uses numba if available, falls back
# to numpy/scipy otherwise.  Callers never need to know about numba.


def affine_transform(input, matrix, offset, output_shape,
                     mode="constant", cval=0.0, order=1):
    """Affine transform — numba-accelerated for 2D/3D when available."""
    if _use_numba_acceleration and input.ndim in (2, 3):
        _ensure_kernels()
        src = np.ascontiguousarray(input, dtype=np.float64)
        mat = np.ascontiguousarray(matrix, dtype=np.float64)
        off = np.ascontiguousarray(offset, dtype=np.float64)
        if input.ndim == 3:
            return _affine_transform_3d(
                src, mat, off,
                output_shape[0], output_shape[1], output_shape[2],
                float(cval), int(order),
            )
        else:
            return _affine_transform_2d(
                src, mat, off,
                output_shape[0], output_shape[1],
                float(cval), int(order),
            )
    from scipy.ndimage import affine_transform as _scipy_affine
    return _scipy_affine(
        input, matrix=matrix, offset=offset, output_shape=output_shape,
        mode=mode, cval=cval, order=order,
    )


def cosine_weights(x):
    """Cosine blending weights — numba-accelerated when available.

    Modifies *x* in-place and returns it.
    """
    if _use_numba_acceleration:
        _ensure_kernels()
        return _cosine_weights_nb(x)
    mask = x < 1
    x[mask] = (np.cos((1 - x[mask]) * np.pi) + 1) / 2
    x = np.clip(x, 0, 1)
    return x


def normalize_weights(weights):
    """Normalize weights along axis 0 — numba-accelerated for 3D/4D.

    Parameters
    ----------
    weights : ndarray of shape (n_views, ...) where ... is 2D or 3D.
    """
    if _use_numba_acceleration and weights.ndim in (3, 4):
        _ensure_kernels()
        w = np.ascontiguousarray(weights, dtype=np.float64)
        if weights.ndim == 4:
            return _normalize_weights_3d_nb(w)
        else:
            return _normalize_weights_2d_nb(w)
    # numpy fallback
    wsum = np.nansum(weights, axis=0)
    wsum[wsum == 0] = 1
    return weights / wsum


def fused_weighted_nansum(images, weights):
    """Fused multiply + nansum — numba-accelerated for 3D/4D.

    Computes ``nansum(images * weights, axis=0)`` in a single pass.
    """
    if _use_numba_acceleration and images.ndim in (3, 4):
        _ensure_kernels()
        im = np.ascontiguousarray(images, dtype=np.float64)
        w = np.ascontiguousarray(weights, dtype=np.float64)
        if images.ndim == 4:
            return _fused_weighted_nansum_3d(im, w)
        else:
            return _fused_weighted_nansum_2d(im, w)
    return np.nansum(images * weights, axis=0)


def nan_to_num(x, nan=0.0):
    """nan_to_num — numba-accelerated for the common nan=0.0 case."""
    if _use_numba_acceleration and nan == 0.0:
        _ensure_kernels()
        return _nan_to_num_nb(np.ascontiguousarray(x, dtype=np.float64))
    return np.nan_to_num(x, nan=nan)
