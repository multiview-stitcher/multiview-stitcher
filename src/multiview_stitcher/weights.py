import dask.array as da
import numpy as np
from scipy.ndimage import gaussian_filter


def calculate_required_overlap(
    method_func=None,
    method_func_kwargs=None,
):
    """
    Calculate the required overlap for fusion given
    - weights method and params
    # - fusion method and params
    """
    if method_func is None:
        return 0
    elif method_func == content_based:
        return 2 * method_func_kwargs["sigma_2"]
    else:
        raise ValueError(f"Unknown weights method {method_func}")


def content_based(
    transformed_sims,
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

    transformed_sims = transformed_sims.astype(np.float32)
    transformed_sims[blending_weights < 1e-7] = np.nan

    weights = [
        nan_gaussian_filter_dask_image(
            (
                sim_t
                - nan_gaussian_filter_dask_image(
                    sim_t, sigma=sigma_1, mode="reflect"
                )
            )
            ** 2,
            sigma=sigma_2,
            mode="reflect",
        )
        for sim_t in transformed_sims
    ]

    weights = np.stack(weights, axis=0)
    weights = normalize_weights(weights)

    return weights


def nan_gaussian_filter_dask_image(ar, *args, **kwargs):
    """
    Gaussian filter ignoring NaNs.

    https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python

    Parameters
    ----------
    ar : da.array

    Returns
    -------
    da.array
        filtered array
    """

    U = ar
    nan_mask = np.isnan(U)
    V = U.copy()
    V[nan_mask] = 0
    VV = gaussian_filter(V, *args, **kwargs)

    W = 0 * U.copy() + 1
    W[nan_mask] = 0
    WW = gaussian_filter(W, *args, **kwargs)

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


def get_smooth_border_weight_from_shape(shape, chunks, widths=None):
    """
    Get a weight image for blending that is 0 at the border and 1 at the center.
    Transition widths can be specified for each dimension.

    20230926: this adds too many small dask tasks, need to optimize

    Possible better strategy:

    Assume that only the border chunks of the input files need to be different
    from da.ones.

    OR: multiscale strategy
    - first fuse smallest scale
    - then go on with only those chunks that need to be filled in
    - not clear if it's beneficial, as it's already known which chunks are needed

    OR: Manually construct output array

    [!!] OR: Use interpolation to calculate distance map
    - reduce input to one value per chunk, i.e. chunksize of [1]*ndim
    - use this to define which input chunk is required for fusion
    - also use this to calculate weights at border
      (only where above coarse interpolation step gave < 1)

    """

    ndim = len(shape)

    # get distance to border for each dim

    dim_dists = [
        da.arange(shape[dim], chunks=chunks[dim]).astype(float)
        for dim in range(ndim)
    ]

    dim_dists = [
        da.min(da.abs(da.stack([dd, dd - shape[idim]])), axis=0)
        for idim, dd in enumerate(dim_dists)
    ]

    dim_ws = [
        smooth_transition(
            dim_dists[dim], x_offset=widths[dim], x_stretch=widths[dim]
        )
        for dim in range(ndim)
    ]

    # get product of weights for each dim
    w = da.ones(shape, chunks=chunks, dtype=float)
    for dim in range(len(shape)):
        tmp_dim_w = dim_ws[dim]
        for _ in range(ndim - dim - 1):
            tmp_dim_w = tmp_dim_w[:, None]
        w *= tmp_dim_w

    return w


def smooth_transition(x, x_offset=0.5, x_stretch=None, k=3):
    """
    Transform the distance from the border to a weight for blending.
    """
    # https://math.stackexchange.com/questions/1832177/sigmoid-function-with-fixed-bounds-and-variable-steepness-partially-solved

    if x_stretch is None:
        x_stretch = x_offset

    xp = x
    xp = xp - x_offset  # w is 0 at offset
    xp = xp / x_stretch / 2.0  # w is +/-0.5 at offset +/- x_stretch

    w = 1 - 1 / (1 + (1 / (xp + 0.5 + 1e-9) - 1) ** (-k))

    # w[xp <= -0.5] = 0.0
    w[xp <= -0.5] = 1e-7
    w[xp >= 0.5] = 1.0

    return w
