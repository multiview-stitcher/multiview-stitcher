from typing import Union

import numpy as np
import xarray as xr
from scipy.fftpack import dctn
from scipy.ndimage import affine_transform, distance_transform_edt, gaussian_filter
from spatial_image import to_spatial_image

from multiview_stitcher import transformation
from multiview_stitcher.misc_utils import clear_cupy_memory, requires_overlap

BoundingBox = dict[str, dict[str, Union[float, int]]]

try:
    import cupy as cp
    import cupyx.scipy.fft as _cupyx_fft
    import cupyx.scipy.ndimage
except ImportError:
    cp = None



@requires_overlap(lambda kwargs: 2 * kwargs["sigma_2"])
def content_based(
    transformed_views,
    blending_weights,
    sigma_1=5,
    sigma_2=11,
):
    """
    Calculate weights for content based fusion, Preibisch implementation.

    W = G_sigma2 * (I - G_sigma1 * I) ** 2

    https://www.researchgate.net/publication/41833243_Mosaicing_of_Single_Plane_Illumination_Microscopy_Images_Using_Groupwise_Registration_and_Fast_Content-Based_Image_Fusion

    Parameters
    ----------
    sims : da.array of dim (n_views, (z,) y, x)
        Input images containing only spatial dimensions.
    params : list of xarray.DataArray
        Transformation parameters for each view.
    sigma_1 : float
        Sigma for inner convolution.
    sigma_2 : float
        Sigma for outer convolution.

    Returns
    -------
    weights : da.array of dim (n_views, (z,) y, x)
        Content based weights for each view.
    """

    transformed_views = transformed_views.astype(np.float32)
    transformed_views[blending_weights < 1e-7] = np.nan

    weights = [
        nan_gaussian_filter(
            (
                sim_t
                - nan_gaussian_filter(
                    sim_t, sigma=sigma_1, mode="reflect"
                )
            )
            ** 2,
            sigma=sigma_2,
            mode="reflect",
        )
        for sim_t in transformed_views
    ]

    weights = np.stack(weights, axis=0)
    weights = normalize_weights(weights)

    return weights


@requires_overlap(
    lambda kwargs: (
        _clamp_overlap(
            kwargs["dct_size"],
            kwargs["output_chunksize"],
        )
    )
)
def content_based_dct(
    transformed_views,
    dct_size=32,
    exponent=1.0,
    otf_support_fraction=0.5,
    output_chunksize=None,
):
    """
    Calculate content-based fusion weights using DCT Shannon entropy.

    The volume is divided into non-overlapping chunks of size ``dct_size``.
    For each chunk the Shannon entropy of the DCT spectrum is computed as the
    quality measure (higher entropy = more structured content):

        H = -sum(|d_i| / dsl1 * log2(|d_i| / dsl1))
        quality = dsl1 * H

    where ``d_i`` are the DCT coefficients and ``dsl1 = sum(|d_i|)``.
    Quality values are placed at chunk centres and then interpolated
    back to the full spatial resolution.

    Reference: Royer et al., "Adaptive light-sheet microscopy for
    long-term, high-resolution imaging in living organisms",
    https://www.nature.com/articles/nbt.3708

    Parameters
    ----------
    transformed_views : np.ndarray of shape (n_views, *spatial_shape)
        Input images containing only spatial dimensions.
    dct_size : int or dict[str, int]
        Chunk size (in pixels) per axis.  A single int is applied
        isotropically; a dict maps spatial dimension names (e.g. "z",
        "y", "x") to per-axis values. Each value is clamped to the
        actual size of that axis. Also defines the required fusion chunk overlap.
    exponent : float
        Exponent to apply to the quality values to increase contrast
        between low and high quality regions.  Default is 1 (no exponent).
    otf_support_fraction : float or None
        Fraction of the per-axis chunk size that lies within the OTF passband.
        Equals ``r_o / N_chunk`` where ``N_chunk = min(dct_sizes)`` and
        ``r_o`` is the L1 frequency-index cutoff used in Royer et al. 2016
        (doi:10.1038/nbt.3708).  Physically, ``otf_support_fraction =
        4 * NA * pixel_size / wavelength`` (all in the same units), so it
        depends only on the imaging system, not on ``dct_size``.  When set,
        ``r_o = otf_support_fraction * min(dct_sizes)`` is computed
        internally; only coefficients with L1 index sum < ``r_o`` contribute
        to the entropy, and the result is scaled by ``2 / r_o^2``.  L2-norm
        normalisation is used.  When ``None`` (default) all coefficients are
        included with L1-mean normalisation.
    output_chunksize : dict or None
        When provided, the required chunk overlap is clamped to this value
        so that it never exceeds the actual processing tile size.

    Returns
    -------
    weights : np.ndarray of shape (n_views, *spatial_shape)
        Normalised content-based weights for each view.
    """

    transformed_views = transformed_views.astype(np.float32)

    if cp is not None and isinstance(transformed_views, cp.ndarray):
        dctn_func = _cupyx_fft.dctn
        affine_transform_func = cupyx.scipy.ndimage.affine_transform
        xp = cp
    else:
        dctn_func = dctn
        affine_transform_func = affine_transform
        xp = np

    spatial_shape = transformed_views.shape[1:]
    ndim = len(spatial_shape)
    sdims = ['z', 'y', 'x'][-ndim:]

    # Normalise dct_size to a per-axis tuple and clamp to spatial_shape (and
    # optionally to output_chunksize so the required overlap doesn't exceed the
    # actual tile size).
    if isinstance(dct_size, dict):
        dct_sizes = tuple(dct_size[d] for d in sdims)
    else:
        dct_sizes = (dct_size,) * ndim
    if output_chunksize is not None:
        dct_sizes = tuple(
            int(min(ds, output_chunksize[dim], s))
            for ds, dim, s in zip(dct_sizes, sdims, spatial_shape)
        )
    else:
        dct_sizes = tuple(
            int(min(ds, s)) for ds, s in zip(dct_sizes, spatial_shape)
        )

    # # make sure dct_size is a factor of the spatial shape to avoid edge effects
    # assert all(
    #     s % dct_sizes[i] == 0
    #     for i, s in enumerate(transformed_views.shape[1:])
    # ), "dct_size must be a factor of the output_chunksize in each dim to avoid edge effects"

    n_chunks = tuple(
        max(1, int(np.ceil(s / dct_sizes[i])))
        for i, s in enumerate(spatial_shape)
    )

    # Pre-allocate quality_maps array to avoid a list + np.stack copy.
    quality_maps = xp.zeros((len(transformed_views),) + n_chunks, dtype=np.float32)

    # Pre-build the L1 frequency-index mask for OTF support radius if requested.
    # r_o scales with chunk size: r_o = otf_support_fraction * min(dct_sizes).
    # The mask is the same shape as each DCT chunk (dct_sizes), so it can be
    # reused across all chunks and views; edge chunks are sliced below.
    if otf_support_fraction is not None:
        _r_o = otf_support_fraction * min(dct_sizes)
        _freq_idx = xp.indices(dct_sizes)                      # (ndim, *dct_sizes)
        _l1_dist = xp.sum(_freq_idx, axis=0)                   # (*dct_sizes,)
        _otf_mask = _l1_dist < _r_o                            # boolean, (*dct_sizes,)
    else:
        _r_o = None
        _otf_mask = None

    for iv, view in enumerate(transformed_views):
        for chunk_idx in np.ndindex(n_chunks):
            slices = tuple(
                slice(
                    ci * dct_sizes[i],
                    min((ci + 1) * dct_sizes[i], spatial_shape[i]),
                )
                for i, ci in enumerate(chunk_idx)
            )
            # Work on a view first; only copy when NaN-filling is needed.
            chunk = view[slices]
            nan_mask = xp.isnan(chunk)
            n_valid = int(xp.sum(~nan_mask))

            # Skip chunks that are mostly invalid.
            if n_valid < 0.2 * chunk.size:
                continue  # quality_maps already zero-initialised

            if nan_mask.any():
                chunk = chunk.copy()
                fill_val = float(xp.nanmin(chunk))
                chunk[nan_mask] = fill_val if fill_val > 0.0001 else 0.0

            d = dctn_func(chunk, norm="ortho")

            if _otf_mask is not None:
                # Slice the pre-built mask to match this (possibly smaller) chunk.
                chunk_slices = tuple(slice(0, s) for s in d.shape)
                mask = _otf_mask[chunk_slices]

                l2_norm = float(xp.sqrt(xp.sum(d ** 2)))
                if l2_norm == 0.0:
                    continue

                p = xp.abs(d[mask]) / l2_norm
                nonzero = p > 0
                entropy = float(-xp.sum(p[nonzero] * xp.log2(p[nonzero])))
                quality_maps[iv][chunk_idx] = (
                    (2.0 / _r_o ** 2) * entropy
                )
                sign = np.sign(quality_maps[iv][chunk_idx])
                quality_maps[iv][chunk_idx] **= exponent
                quality_maps[iv][chunk_idx] *= sign
            else:
                # Reuse the DCT buffer in-place to avoid extra allocations.
                xp.abs(d, out=d)                                # d now holds |coeff|
                dsl1 = float(d.mean())
                if dsl1 == 0.0:
                    continue

                d /= dsl1                                       # d now holds p = |coeff|/dsl1
                d_flat = d.ravel()                              # view (no copy) when C-contiguous
                nonzero = d_flat > 0
                entropy = float(-xp.dot(d_flat[nonzero], xp.log2(d_flat[nonzero])))
                quality_maps[iv][chunk_idx] = (dsl1 * entropy) ** exponent

    quality_maps -= xp.nanmin(quality_maps, axis=0)

    quality_maps = normalize_weights(quality_maps)

    # Interpolate each view's quality map back to the full spatial resolution
    # using affine_transform (order=1).  Write directly into the output array
    # via the `output` kwarg to avoid a temporary full-spatial allocation per view.
    weights = xp.zeros_like(transformed_views)
    # Mapping from output pixel p to quality-map index q:
    #   q = (p - (ds-1)/2) / ds
    # so scale = 1/ds, offset = -(ds-1)/(2*ds).
    scale = tuple(1.0 / ds for ds in dct_sizes)
    matrix = xp.diag(xp.array(scale, dtype=xp.float64))
    offset = tuple(-(ds - 1) / (2.0 * ds) for ds in dct_sizes)
    for i, qmap in enumerate(quality_maps):
        affine_transform_func(
            qmap,
            matrix,
            offset=offset,
            output_shape=spatial_shape,
            order=1,
            mode="nearest",
            output=weights[i],
        )

    weights = normalize_weights(weights)

    if xp is cp:
        del quality_maps
        clear_cupy_memory()

    return weights


def nan_gaussian_filter(ar, *args, **kwargs):
    """
    Gaussian filter ignoring NaNs.

    https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
    """

    if cp is not None and isinstance(ar, cp.ndarray):
        gaussian_filter_func = cupyx.scipy.ndimage.gaussian_filter
    else:
        gaussian_filter_func = gaussian_filter

    U = ar
    nan_mask = np.isnan(U)
    V = U.copy()
    V[nan_mask] = 0
    VV = gaussian_filter_func(V, *args, **kwargs)

    W = 0 * U.copy() + 1
    W[nan_mask] = 0
    WW = gaussian_filter_func(W, *args, **kwargs)

    # avoid division by zero
    WW[nan_mask] = 1

    Z = VV / WW

    Z[nan_mask] = np.nan

    return Z


def normalize_weights(weights):
    """
    Normalize weights.

    Parameters
    ----------
    weights : da.array of dim (n_views, (z,) y, x)
        Weights to normalize.

    Returns
    -------
    weights : da.array of dim (n_views, (z,) y, x)
        Normalized weights.
    """

    wsum = np.nansum(weights, axis=0)
    wsum[wsum == 0] = 1

    weights = weights / wsum

    return weights


def _shrink_source_bb(
    source_bb: BoundingBox,
    shrink_distance: Union[float, dict[str, float]],
) -> BoundingBox:
    """
    Return a copy of ``source_bb`` whose physical extent has been reduced by
    ``shrink_distance`` on *each* side along every spatial dimension.

    Parameters
    ----------
    source_bb :
        Original source bounding box.
    shrink_distance :
        Physical shrinkage per side.  A single float is applied isotropically;
        a dict maps dimension names to individual values.

    Returns
    -------
    BoundingBox
        New bounding box with ``origin`` shifted inward and ``shape`` reduced
        so that the physical extent shrinks by ``2 * shrink_distance[dim]``
        along each dimension.
    """

    sdims = list(source_bb["origin"].keys())
    if isinstance(shrink_distance, (int, float)):
        shrink_distance = {dim: float(shrink_distance) for dim in sdims}

    return {
        "origin": {
            dim: source_bb["origin"][dim] + shrink_distance.get(dim, 0)
            for dim in sdims
        },
        "spacing": source_bb["spacing"],
        "shape": {
            dim: source_bb["shape"][dim]
            - 2 * shrink_distance.get(dim, 0) / source_bb["spacing"][dim]
            for dim in sdims
        },
    }


def get_blending_weights(
    target_bb: BoundingBox,
    source_bb: BoundingBox,
    affine: xr.DataArray,
    blending_widths: dict[str, float] = None,
    shrink_distance: float = 0,
    cupy=False,
):
    """
    Calculate smooth blending weights for fusion.

    Note: The resulting weights
    - are not normalized
    - are not pixel perfect at the edges and need to be restricted
      to valid regions in the target space (done in fusion function)
    - should be non-zero for all valid pixels in the target space

    Parameters
    ----------
    target_bb : Target bounding box.
    source_bb : Source bounding box.
    affine : xr.DataArray
        Affine transformation from source to target space.
    blending_widths : dict, optional
        Physical blending widths for each dimension.
    shrink_distance : float or dict[str, float], optional
        Shrink the source bounding box inward by this many physical units on
        each side before computing the weights.  A single float is applied
        isotropically; a dict maps dimension names to individual values.
        Defaults to 0 (no shrinkage).  Use this to make weights reach zero
        *before* the actual view border, e.g. to avoid border artefacts in
        convolution-based fusion methods such as multi-view deconvolution.

    Returns
    -------
    target_weights : dask array containing blending weights
        for the target bounding box
    """

    if blending_widths is None:
        blending_widths = {"z": 3, "y": 10, "x": 10}

    sdims = sorted(source_bb["origin"].keys())[::-1]

    if shrink_distance:
        source_bb = _shrink_source_bb(source_bb, shrink_distance)
    ndim = len(sdims)

    mask = np.zeros([3 + 2 for dim in sdims])
    mask[(slice(1, -1),) * ndim] = 1
    support_spacing = {
        dim: (source_bb["shape"][dim] - 1) / 4 * source_bb["spacing"][dim]
        for dim in sdims
    }

    # slightly enlargen the support to avoid edge effects
    # otherwise there's no smooth transition at shared coordinate boundaries
    edt_support_spacing = {
        dim: support_spacing[dim]
        * (source_bb["shape"][dim] - 1 + 2 * 1)
        / (source_bb["shape"][dim] - 1)
        for dim in sdims
    }
    edt_support_origin = {
        dim: source_bb["origin"][dim] - 1 * source_bb["spacing"][dim]
        for dim in sdims
    }

    edt_support = distance_transform_edt(
        mask,
        sampling=[
            edt_support_spacing[dim] / blending_widths[dim] for dim in sdims
        ],
    )
    edt_support = to_spatial_image(
        edt_support,
        scale=edt_support_spacing,
        translation=edt_support_origin,
    )

    if cp is not None and cupy:
        edt_support.data = cp.asarray(edt_support.data)

    target_weights = transformation.transform_sim(
        edt_support.astype(np.float32),
        p=np.linalg.inv(affine),
        output_stack_properties=target_bb,
        order=1,
        cval=0.0,
    )

    ## Note: Restriction to valid regions in the target space
    ## is done in the fusion function.

    # support = to_spatial_image(
    #     mask,
    #     scale=support_spacing,
    #     translation=support_origin,
    # )

    # target_support = transformation.transform_sim(
    #     support.astype(np.float32),
    #     p=np.linalg.inv(affine),
    #     output_stack_properties=target_bb,
    #     order=0,
    #     cval=np.nan,
    # )

    # target_weights = target_weights * ~np.isnan(target_support)

    def cosine_weights(x):
        mask = x < 1
        x[mask] = (np.cos((1 - x[mask]) * np.pi) + 1) / 2
        x = np.clip(x, 0, 1)
        # x[~mask] = 1
        return x

    target_weights.data = cosine_weights(target_weights.data)

    return target_weights.data


def _clamp_overlap(overlap, output_chunksize):
    """Clamp overlap to output_chunksize when it is provided."""

    # normalize overlap to dict[str, int]
    sdims = sorted(output_chunksize.keys())[::-1]
    if not isinstance(overlap, dict):
        overlap = {dim: int(overlap) for dim in sdims}

    return {
        dim: min(overlap[dim], output_chunksize[dim]) for dim in sdims
    }
