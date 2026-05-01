"""
Bayesian multi-view deconvolution fusion.

Implements the efficient Bayesian-based multiview deconvolution method from:

    Preibisch et al., "Efficient Bayesian-based multiview deconvolution",
    Nature Methods 11, 645–648 (2014). https://doi.org/10.1038/nmeth.2929

Algorithm (sequential per-view update):
    Initialise *ψ* as a blending-weighted average of the input views.
    For each iteration:
        For each view v:
            1. Forward:     blurred  = convolve(ψ, PSF_v)
            2. Quotient:    ratio    = img_v / blurred  (1 where no image data)
            3. Back-project: integral = convolve(ratio, kernel2_v)
            4. Update:      ψ ← ψ + (clamp(ψ · integral) − ψ) · w_v

The back-projection kernel (``kernel2``) is a compound kernel derived from all
view PSFs.  Four variants are provided (see :class:`PSFType`).

Based on the BigStitcher / multiview-reconstruction implementation:
    https://github.com/JaneliaSciComp/multiview-reconstruction
"""

from enum import Enum

import numpy as np
from scipy.ndimage import convolve as _scipy_convolve
from scipy.ndimage import gaussian_filter as _scipy_gaussian_filter

try:
    import cupy as cp
    import cupyx.scipy.ndimage as _cupyx_ndimage
except ImportError:
    cp = None


# ---------------------------------------------------------------------------
# PSF type enum
# ---------------------------------------------------------------------------

class PSFType(str, Enum):
    """Compound back-projection kernel type.

    Controls how the back-projection kernel ``kernel2`` is derived from the
    per-view PSFs.  See Preibisch et al. 2014 (Supplementary) for derivations.
    """

    EFFICIENT_BAYESIAN = "EFFICIENT_BAYESIAN"
    """Full compound kernel (most accurate; recommended default):

    ``kernel2_v = normalise( flip(PSF_v) · ∏_{w≠v} (flip(PSF_v) ⊛ PSF_w ⊛ flip(PSF_w)) )``
    """

    OPTIMIZATION_I = "OPTIMIZATION_I"
    """Simplified compound kernel (faster):

    ``kernel2_v = flip( normalise( PSF_v · ∏_{w≠v} (flip(PSF_v) ⊛ PSF_w) ) )``
    """

    OPTIMIZATION_II = "OPTIMIZATION_II"
    """Exponential kernel (fastest):

    ``kernel2_v = flip( normalise( PSF_v ^ n_views ) )``
    """

    INDEPENDENT = "INDEPENDENT"
    """Standard Richardson-Lucy per view; no cross-view coupling:

    ``kernel2_v = flip(PSF_v)``
    """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _convolve(arr, kernel, mode="mirror", cval=0.0):
    """Convolve *arr* with *kernel*, dispatching to cupy when appropriate.

    Kernels may be numpy arrays even when *arr* is a cupy array; the
    conversion is handled here so callers need not think about devices.
    """
    if cp is not None and isinstance(arr, cp.ndarray):
        return _cupyx_ndimage.convolve(
            arr, cp.asarray(kernel), mode=mode, cval=cval
        )
    return _scipy_convolve(arr, kernel, mode=mode, cval=cval)


def _flip(kernel):
    """180-degree rotation of *kernel* (flip all axes).  CPU-only (numpy)."""
    return np.flip(kernel)


def _norm(kernel):
    """Normalise *kernel* so it sums to 1.  Returns float32 numpy array."""
    kernel = np.asarray(kernel, dtype=np.float64)
    s = kernel.sum()
    if s > 0:
        kernel = kernel / s
    return kernel.astype(np.float32)


# ---------------------------------------------------------------------------
# Public PSF helpers
# ---------------------------------------------------------------------------

def make_gaussian_psf(sigma, ndim=None, shape=None):
    """Create a normalised isotropic or anisotropic Gaussian PSF.

    Parameters
    ----------
    sigma : float or sequence of float
        Standard deviation(s) in pixels.  A scalar means isotropic.
        When a sequence, one value per spatial dimension in
        **(z, y, x)** order for 3-D data.
    ndim : int, optional
        Number of spatial dimensions.  Only required when *sigma* is a scalar.
    shape : tuple of int, optional
        Kernel shape.  Defaults to ``ceil(6·σ) | 1`` per dimension
        (rounded up to the next odd integer).

    Returns
    -------
    np.ndarray, float32
        Normalised Gaussian PSF.
    """
    sigma = np.atleast_1d(np.asarray(sigma, dtype=float))
    if sigma.size == 1 and ndim is not None:
        sigma = np.full(ndim, float(sigma[0]))

    if shape is None:
        # ceil(6σ), then force odd
        shape = tuple(int(np.ceil(6.0 * s)) | 1 for s in sigma)

    psf = np.zeros(shape, dtype=np.float32)
    psf[tuple(s // 2 for s in shape)] = 1.0  # Kronecker delta at centre
    psf = _scipy_gaussian_filter(psf, sigma=sigma.tolist())
    return _norm(psf)


def estimate_psf(spacing, na=0.8, wavelength_um=0.5):
    """Estimate a Gaussian PSF from pixel spacing and objective parameters.

    Uses Rayleigh-criterion half-widths converted to Gaussian sigma:

    * **Lateral (y, x):** σ ≈ 0.5 · λ / NA  (in physical units → pixels)
    * **Axial   (z):**    σ ≈ 2.0 · λ / NA²  (in physical units → pixels)

    Parameters
    ----------
    spacing : dict
        Pixel spacing in physical units (e.g. µm) for each spatial dimension.
        Keys are dimension names (``'z'``, ``'y'``, ``'x'``).
    na : float, optional
        Numerical aperture of the detection objective.  Default ``0.8``.
    wavelength_um : float, optional
        Emission wavelength in µm.  Default ``0.5`` (green light).

    Returns
    -------
    np.ndarray, float32
        Normalised Gaussian PSF.
    """
    sigma_lateral = 0.5 * wavelength_um / na
    sigma_axial = 2.0 * wavelength_um / (na ** 2)

    sigma_px = []
    for dim, sp in spacing.items():
        if dim == "z":
            sigma_px.append(max(0.5, sigma_axial / float(sp)))
        else:
            sigma_px.append(max(0.5, sigma_lateral / float(sp)))

    return make_gaussian_psf(sigma_px)


# ---------------------------------------------------------------------------
# Compound back-projection kernel computation  (CPU only)
# ---------------------------------------------------------------------------

def _compute_compound_kernel(v_idx, psfs, psf_type):
    """Compute the back-projection kernel2 for view *v_idx*.

    All calculations are performed on the CPU with numpy/scipy regardless of
    where the image data lives; the result is moved to the target device in
    :func:`bayesian_fusion`.

    Parameters
    ----------
    v_idx : int
        Index of the current view in *psfs*.
    psfs : list of np.ndarray
        Normalised PSF (kernel1) for each view, all with the same shape.
    psf_type : str or PSFType
        Compound kernel variant.

    Returns
    -------
    np.ndarray, float32
        The back-projection kernel2 for view *v_idx*.
    """
    n_views = len(psfs)
    psf_type = str(psf_type)  # accept both str and PSFType enum
    psf_v = psfs[v_idx].astype(np.float64)

    if n_views == 1 or psf_type == PSFType.INDEPENDENT:
        # Standard Richardson-Lucy: kernel2 = flip(PSF_v)
        return _norm(_flip(psf_v))

    if psf_type == PSFType.OPTIMIZATION_II:
        # kernel2 = flip( normalise( PSF_v ^ n_views ) )
        return _norm(_flip(psf_v ** n_views))

    if psf_type == PSFType.OPTIMIZATION_I:
        # kernel2 = flip( normalise( PSF_v · ∏_{w≠v} (flip(PSF_v) ⊛ PSF_w) ) )
        tmp = psf_v.copy()
        psf_v_flip = _flip(psf_v)
        for w_idx, psf_w in enumerate(psfs):
            if w_idx == v_idx:
                continue
            conv = _scipy_convolve(
                psf_v_flip, psf_w.astype(np.float64),
                mode="constant", cval=0.0,
            )
            tmp = tmp * conv
        return _norm(_flip(tmp))

    # EFFICIENT_BAYESIAN (default)
    # kernel2_v = normalise( flip(PSF_v) · ∏_{w≠v} (flip(PSF_v) ⊛ PSF_w ⊛ flip(PSF_w)) )
    psf_v_flip = _flip(psf_v)
    tmp = psf_v_flip.copy()
    for w_idx, psf_w in enumerate(psfs):
        if w_idx == v_idx:
            continue
        psf_w_d = psf_w.astype(np.float64)
        # flip(PSF_v) ⊛ PSF_w  (output same shape as flip(PSF_v))
        conv1 = _scipy_convolve(
            psf_v_flip, psf_w_d,
            mode="constant", cval=0.0,
        )
        # (flip(PSF_v) ⊛ PSF_w) ⊛ flip(PSF_w)
        conv2 = _scipy_convolve(
            conv1, _flip(psf_w_d),
            mode="constant", cval=0.0,
        )
        tmp = tmp * conv2
    return _norm(tmp)


# ---------------------------------------------------------------------------
# Main fusion function
# ---------------------------------------------------------------------------

def bayesian_fusion(
    transformed_views,
    blending_weights,
    psfs=None,
    psf_type=PSFType.EFFICIENT_BAYESIAN,
    n_iterations=10,
    lambda_reg=0.0,
    min_value=1e-4,
    psf_sigma_px=None,
    output_spacing=None,
    na=0.8,
    wavelength_um=0.5,
):
    """Bayesian multi-view deconvolution fusion.

    Fuses multiple pre-transformed views via iterative Richardson-Lucy
    deconvolution with per-view compound back-projection kernels.  Blending
    weights gate the update at each pixel so that only views that
    genuinely contribute are used.

    This function follows the signature convention of the built-in fusion
    functions and can be passed directly as ``fusion_func`` to
    :func:`multiview_stitcher.fusion.fuse`.

    Parameters
    ----------
    transformed_views : np.ndarray or cp.ndarray, shape (n_views, [z,] y, x)
        Pre-transformed views in the output space.
        ``NaN`` marks pixels outside a view's field of view.
    blending_weights : np.ndarray or cp.ndarray, shape (n_views, [z,] y, x)
        Per-pixel blending weights (normalised, sum ≤ 1 per pixel across views).
        Typically produced by :func:`multiview_stitcher.weights.get_blending_weights`.
    psfs : list of np.ndarray, optional
        Explicit PSF for each view (len == n_views).  Each PSF must have
        *ndim* dimensions matching the spatial data.  PSFs are normalised
        internally.  If ``None``, a Gaussian PSF is estimated (see below).
    psf_type : str or PSFType, optional
        Compound back-projection kernel variant.  One of:

        * ``"EFFICIENT_BAYESIAN"`` – full compound kernel (default, most accurate)
        * ``"OPTIMIZATION_I"`` – simplified compound kernel (faster)
        * ``"OPTIMIZATION_II"`` – exponential kernel (fastest)
        * ``"INDEPENDENT"`` – standard RL per view (no cross-view coupling)

        By default :attr:`PSFType.EFFICIENT_BAYESIAN`.
    n_iterations : int, optional
        Number of deconvolution iterations.  By default ``10``.
    lambda_reg : float, optional
        Tikhonov regularisation strength.  ``0`` disables regularisation
        (default).  Values around ``1e-4`` – ``1e-2`` provide mild smoothing.
    min_value : float, optional
        Minimum pixel value for the deconvolved image and denominators.
        By default ``1e-4``.
    psf_sigma_px : float or sequence of float, optional
        Gaussian PSF sigma(s) in pixels, used when *psfs* is ``None``.
        Scalar → isotropic; sequence → one value per spatial dimension
        in **(z, y, x)** order.  Defaults to ``1.5`` pixels isotropic when
        neither *psfs* nor *psf_sigma_px* are given and *output_spacing* is
        also absent.
    output_spacing : dict, optional
        Pixel spacing in physical units (µm) keyed by dimension name, used to
        estimate a physically motivated PSF when *psfs* and *psf_sigma_px* are
        both ``None``.  Requires *na* and *wavelength_um*.
    na : float, optional
        Numerical aperture, used together with *output_spacing* for PSF
        estimation.  Default ``0.8``.
    wavelength_um : float, optional
        Emission wavelength in µm, used together with *output_spacing*.
        Default ``0.5`` (500 nm, green channel).

    Returns
    -------
    np.ndarray or cp.ndarray, same dtype as *transformed_views*
        Fused (deconvolved) image of shape ``([z,] y, x)``.

    Notes
    -----
    When *transformed_views* is a **cupy** array the convolutions are executed
    on the GPU via ``cupyx.scipy.ndimage``.  The compound kernels are always
    computed on the CPU and transferred to the device at the start of the
    fusion call.
    """
    n_views = transformed_views.shape[0]
    ndim = transformed_views.ndim - 1
    input_dtype = transformed_views.dtype

    # ------------------------------------------------------------------
    # Replace NaN → 0 in observed images (0 = "no data" sentinel)
    # ------------------------------------------------------------------
    observed = np.nan_to_num(
        transformed_views.astype(np.float32), nan=0.0
    )

    # ------------------------------------------------------------------
    # Build / validate PSFs
    # ------------------------------------------------------------------
    if psfs is None:
        if psf_sigma_px is not None:
            psf0 = make_gaussian_psf(psf_sigma_px, ndim=ndim)
        elif output_spacing is not None:
            psf0 = estimate_psf(output_spacing, na=na, wavelength_um=wavelength_um)
        else:
            psf0 = make_gaussian_psf(1.5, ndim=ndim)
        psfs_cpu = [psf0] * n_views
    else:
        if len(psfs) != n_views:
            raise ValueError(
                f"len(psfs) = {len(psfs)}, but n_views = {n_views}. "
                "Provide one PSF per view."
            )
        psfs_cpu = [_norm(np.asarray(p, dtype=np.float32)) for p in psfs]

    # Ensure all PSFs have the same shape (pad with zeros to the maximum
    # extent along each dimension)
    max_shape = tuple(
        max(p.shape[d] for p in psfs_cpu) for d in range(ndim)
    )
    padded = []
    for p in psfs_cpu:
        if p.shape != max_shape:
            pad_widths = []
            for a, t in zip(p.shape, max_shape):
                diff = t - a
                pad_widths.append((diff // 2, diff - diff // 2))
            p = np.pad(p, pad_widths, mode="constant")
        padded.append(_norm(p))
    psfs_cpu = padded

    # ------------------------------------------------------------------
    # Compute compound back-projection kernels (always on CPU)
    # ------------------------------------------------------------------
    # kernels stay as numpy arrays; _convolve handles device transfer
    kernels1 = psfs_cpu
    kernels2 = [
        _compute_compound_kernel(v, psfs_cpu, psf_type)
        for v in range(n_views)
    ]

    # ------------------------------------------------------------------
    # Initialisation: blending-weighted average of observed views
    # ------------------------------------------------------------------
    # blending_weights are already normalised (sum ≤ 1 per pixel), so the
    # sum-to-1 denominator is effectively already applied.
    psi = np.nansum(observed * blending_weights, axis=0).astype(np.float32)
    psi = psi.clip(np.float32(min_value))

    max_intensity = float(psi.max())
    if max_intensity <= 0:
        max_intensity = 1.0

    # ------------------------------------------------------------------
    # Iterative deconvolution (sequential per-view update)
    # ------------------------------------------------------------------
    for _it in range(n_iterations):
        for v in range(n_views):
            w_v = blending_weights[v]    # shape: spatial
            img_v = observed[v]          # shape: spatial

            # 1. Forward projection: blur current estimate with PSF_v
            #    Boundary: mirror extension (avoids edge ringing)
            blurred = _convolve(psi, kernels1[v], mode="mirror")

            # 2. Quotient: img_v / blurred
            #    Where img_v == 0 (no data), set ratio to 1 (no update)
            ratio = np.where(
                img_v > 0.0,
                img_v / np.maximum(blurred, np.float32(min_value)),
                np.ones_like(blurred),
            )

            # 3. Back-projection: convolve ratio with compound kernel
            #    Boundary: constant=1 (outside psi → no correction from there)
            integral = _convolve(ratio, kernels2[v], mode="constant", cval=1.0)

            # 4. Update step
            value = psi * integral

            if lambda_reg > 0:
                # Tikhonov regularisation:
                #   f(x) = (sqrt(1 + 2·λ·x) − 1) / λ
                # applied after normalising to [0, 1] range
                x = np.maximum(value, np.float32(0.0)) / max_intensity
                adjusted = (
                    np.sqrt(
                        np.float32(1.0) + np.float32(2.0 * lambda_reg) * x
                    ) - np.float32(1.0)
                ) / np.float32(lambda_reg) * max_intensity
            else:
                adjusted = value

            next_psi = np.where(
                np.isnan(adjusted),
                np.float32(min_value),
                np.maximum(adjusted, np.float32(min_value)),
            )

            # Weighted update: damp correction by blending weight w_v.
            # Where w_v = 0 (view does not cover this pixel), psi is unchanged.
            psi = psi + (next_psi - psi) * w_v

    return psi.astype(input_dtype)
