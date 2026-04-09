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
    (``distance_transform_edt``, ``gaussian_filter``) fall back to CPU.

    Phase cross-correlation and structural similarity are implemented
    natively in JAX using FFTs and sliding-window reductions.  The masked
    variant of phase_cross_correlation falls back to CPU skimage because
    it uses a fundamentally different (normalised cross-correlation)
    algorithm.
    """

    _has_native_affine_transform = True
    _has_native_phase_cross_correlation = True
    _has_native_structural_similarity = True
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

    # -- Native phase cross-correlation via JAX FFT ----------------------------

    def _native_phase_cross_correlation(self, reference_image, moving_image, **kwargs):
        """FFT-based phase cross-correlation running entirely on the JAX device.

        The masked variant (``reference_mask`` / ``moving_mask`` kwargs) uses a
        different algorithm and is delegated to CPU skimage.
        """
        jnp = self.xp

        if "reference_mask" in kwargs or "moving_mask" in kwargs:
            return self._skimage_phase_cross_correlation(
                reference_image, moving_image, **kwargs
            )

        normalization = kwargs.get("normalization", "phase")
        upsample_factor = int(kwargs.get("upsample_factor", 1))

        src_freq = jnp.fft.fftn(reference_image)
        target_freq = jnp.fft.fftn(moving_image)

        image_product = src_freq * jnp.conj(target_freq)

        if normalization == "phase":
            eps = jnp.finfo(image_product.real.dtype).eps
            image_product = image_product / (jnp.abs(image_product) + eps)
        elif normalization is not None:
            raise ValueError(
                f"normalization must be 'phase' or None, got {normalization!r}"
            )

        cross_correlation = jnp.real(jnp.fft.ifftn(image_product))

        # Integer-pixel peak
        maxima_idx = int(jnp.argmax(jnp.abs(cross_correlation)))
        shape = cross_correlation.shape
        maxima = np.array(np.unravel_index(maxima_idx, shape))
        midpoints = np.array([s // 2 for s in shape])
        shift = np.where(
            maxima > midpoints, maxima - np.array(shape), maxima
        ).astype(float)

        if upsample_factor == 1:
            return (jnp.array(shift), 0.0, 0.0)

        # Sub-pixel refinement via upsampled matrix DFT
        upsample_factor_f = float(upsample_factor)
        shift = np.round(shift * upsample_factor_f) / upsample_factor_f
        upsampled_region_size = int(np.ceil(upsample_factor_f * 1.5))
        dftshift = np.fix(upsampled_region_size / 2.0)
        sample_region_offset = dftshift - shift * upsample_factor_f

        cross_correlation_up = self._jax_upsampled_dft(
            jnp.conj(image_product),
            upsampled_region_size,
            upsample_factor_f,
            sample_region_offset,
        ) / (cross_correlation.size * upsample_factor_f ** reference_image.ndim)

        maxima_idx_up = int(jnp.argmax(jnp.abs(cross_correlation_up)))
        maxima_up = np.array(
            np.unravel_index(maxima_idx_up, cross_correlation_up.shape)
        ).astype(float)
        maxima_up -= dftshift

        shift = shift + maxima_up / upsample_factor_f
        return (jnp.array(shift), 0.0, 0.0)

    def _jax_upsampled_dft(self, data, upsampled_region_size, upsample_factor, axis_offsets):
        """Matrix-DFT based upsampled DFT for sub-pixel shift estimation.

        Port of skimage's ``_upsampled_dft`` to JAX
        (Guizar-Sicairos et al., Optics Letters, 2008).
        """
        jnp = self.xp
        im2pi = 1j * 2 * np.pi

        dims = data.shape
        if not hasattr(upsampled_region_size, "__len__"):
            upsampled_region_size = [int(upsampled_region_size)] * len(dims)
        if not hasattr(axis_offsets, "__len__"):
            axis_offsets = [float(axis_offsets)] * len(dims)

        for n_items, ups_size, ax_off in reversed(
            list(zip(dims, upsampled_region_size, axis_offsets))
        ):
            freq = np.fft.ifftshift(np.arange(n_items)) - np.floor(n_items / 2.0)
            samples = np.arange(ups_size) - ax_off
            kernel = jnp.array(
                np.exp(
                    (-im2pi / (n_items * upsample_factor))
                    * np.outer(freq, samples)
                )
            )
            # kernel shape: (n_items, ups_size) → kernel.T: (ups_size, n_items)
            # contract data's last dim with kernel.T's last dim
            data = jnp.tensordot(kernel.T, data, axes=([-1], [-1]))

        return data

    # -- Native structural similarity via JAX uniform filter ------------------

    def _native_structural_similarity(self, im1, im2, **kwargs):
        """SSIM computed entirely on the JAX device.

        Uses a uniform sliding-window filter implemented via
        ``jax.lax.reduce_window`` with reflect padding, matching
        scipy's ``uniform_filter(mode='reflect')`` behaviour.
        """
        import jax.lax as lax

        jnp = self.xp

        data_range = float(kwargs.get("data_range", float(jnp.max(im2) - jnp.min(im2))))
        win_size = int(kwargs.get("win_size", 7))
        K1 = float(kwargs.get("K1", 0.01))
        K2 = float(kwargs.get("K2", 0.03))

        C1 = (K1 * data_range) ** 2
        C2 = (K2 * data_range) ** 2
        ndim = im1.ndim
        NP = win_size ** ndim
        # Bessel's correction for sample covariance, matching skimage's default
        # (use_sample_covariance=True)
        cov_norm = NP / (NP - 1)

        im1 = im1.astype(jnp.float32)
        im2 = im2.astype(jnp.float32)
        pad = win_size // 2

        def uniform_filter(x):
            # scipy.ndimage.uniform_filter applies a 1-D box along each axis in
            # sequence (separable), with mode='reflect' which is edge-inclusive
            # (= numpy/JAX 'symmetric').  Reproducing the separable approach
            # is essential: a single N-D reduce_window gives different boundary
            # values at corners because 2-D padding interacts differently than
            # two successive 1-D paddings.
            for axis in range(ndim):
                pad_cfg = [(0, 0)] * ndim
                pad_cfg[axis] = (pad, pad)
                x = lax.reduce_window(
                    jnp.pad(x, pad_cfg, mode="symmetric"),
                    init_value=0.0,
                    computation=lax.add,
                    window_dimensions=tuple(
                        win_size if d == axis else 1 for d in range(ndim)
                    ),
                    window_strides=(1,) * ndim,
                    padding="VALID",
                ) / win_size
            return x

        ux = uniform_filter(im1)
        uy = uniform_filter(im2)
        uxx = uniform_filter(im1 * im1)
        uyy = uniform_filter(im2 * im2)
        uxy = uniform_filter(im1 * im2)

        vx = cov_norm * (uxx - ux * ux)
        vy = cov_norm * (uyy - uy * uy)
        vxy = cov_norm * (uxy - ux * uy)

        A1 = 2.0 * ux * uy + C1
        A2 = 2.0 * vxy + C2
        B1 = ux ** 2 + uy ** 2 + C1
        B2 = vx + vy + C2

        S = (A1 * A2) / (B1 * B2)

        # Crop the filter-radius border to match skimage's behaviour
        # (skimage: `crop(S, pad).mean(dtype=float64)`)
        pad_crop = (win_size - 1) // 2
        if pad_crop > 0:
            S = S[tuple(slice(pad_crop, -pad_crop) for _ in range(ndim))]
        return jnp.mean(S)

    def __repr__(self):
        return "JaxBackend()"
