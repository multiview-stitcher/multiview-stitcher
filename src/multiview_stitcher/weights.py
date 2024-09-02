from typing import Union

import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt, gaussian_filter
from spatial_image import to_spatial_image

from multiview_stitcher import transformation

BoundingBox = dict[str, dict[str, Union[float, int]]]


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


def get_blending_weights(
    target_bb: BoundingBox,
    source_bb: BoundingBox,
    affine: xr.DataArray,
    blending_widths: dict[str, float] = None,
):
    """
    Calculate smooth blending weights for fusion.

    Parameters
    ----------
    target_bb : Target bounding box.
    source_bb : Source bounding box.
    params : list of xarray.DataArray
        Transformation parameters for each view.
    blending_widths : dict
        Physical blending widths for each dimension.

    Returns
    -------
    target_weights : spatial_image
    """

    if blending_widths is None:
        blending_widths = {"z": 3, "y": 10, "x": 10}

    sdims = sorted(source_bb["origin"].keys())[::-1]
    ndim = len(sdims)

    mask = np.zeros([3 + 2 for dim in sdims])
    mask[(slice(1, -1),) * ndim] = 1
    support_spacing = {
        dim: (source_bb["shape"][dim] - 1) / 4 * source_bb["spacing"][dim]
        for dim in sdims
    }
    support_origin = source_bb["origin"]

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

    target_weights = transformation.transform_sim(
        edt_support.astype(np.float32),
        p=np.linalg.inv(affine),
        output_stack_properties=target_bb,
        order=1,
        cval=np.nan,
    )

    support = to_spatial_image(
        mask,
        scale=support_spacing,
        translation=support_origin,
    )

    target_support = transformation.transform_sim(
        support.astype(np.float32),
        p=np.linalg.inv(affine),
        output_stack_properties=target_bb,
        order=0,
        cval=np.nan,
    )

    target_weights = target_weights * ~np.isnan(target_support)

    def cosine_weights(x):
        mask = x < 1
        x[mask] = (np.cos((1 - x[mask]) * np.pi) + 1) / 2
        x = np.clip(x, 0, 1)
        # x[~mask] = 1
        return x

    target_weights.data = cosine_weights(target_weights.data)

    return target_weights
