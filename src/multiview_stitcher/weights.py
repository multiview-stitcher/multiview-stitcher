import dask.array as da
from dask_image.ndfilters import gaussian_filter


def calc_content_based_weights(
    transformed_sims,
    params=None,
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

    weights = [
        gaussian_filter(
            (sim_t - gaussian_filter(sim_t, sigma=sigma_1)) ** 2, sigma=sigma_2
        )
        for sim_t in transformed_sims
    ]

    weights = da.stack(weights, axis=0)

    return weights


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

    wsum = da.nansum(weights, axis=0)
    wsum[wsum == 0] = 1

    weights = weights / wsum

    return weights
