from typing import Union

import numpy as np
import xarray as xr
from numba import njit
from scipy.ndimage import gaussian_filter

from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher.geometry import interior_distance

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
        for sim_t in transformed_views
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

    # avoid division by zero
    WW[nan_mask] = 1

    Z = VV / WW

    Z[nan_mask] = np.nan

    return Z


@njit
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

    wsum = np.zeros(weights.shape[1:])
    for index in np.ndindex(weights.shape[1:]):
        for iview in range(weights.shape[0]):
            if not np.isnan(weights[iview][index]):
                wsum[index] += weights[iview][index]

        if wsum[index] == 0:
            wsum[index] = 1

    weights = weights / wsum

    return weights


def get_blending_weights(
    target_bb: BoundingBox,
    source_bbs: list[BoundingBox],
    affines: list[xr.DataArray],
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

    sdims = si_utils.get_spatial_dims_from_stack_properties(target_bb)
    len(sdims)
    nviews = len(source_bbs)

    pts = sim_meshgrid(
        shape=tuple([target_bb["shape"][dim] for dim in sdims]),
        spacing=tuple([target_bb["spacing"][dim] for dim in sdims]),
        origin=tuple([target_bb["origin"][dim] for dim in sdims]),
    )

    target_weights = np.zeros(
        (nviews,) + tuple(target_bb["shape"][dim] for dim in sdims)
    )
    for iview, source_bb in enumerate(source_bbs):
        target_weights[iview] = interior_distance(
            pts,
            source_bb,
            affines[iview],
            dim_factors={dim: 1 / blending_widths[dim] for dim in sdims},
        ).reshape([target_bb["shape"][dim] for dim in sdims])

    target_weights = cosine_weights(target_weights)

    target_weights = normalize_weights(target_weights)

    return target_weights


@njit
def cosine_weights(x):
    for index in np.ndindex(x.shape):
        if x[index] < 1:
            x[index] = (np.cos((1 - x[index]) * np.pi) + 1) / 2
        else:
            x[index] = 1
    return x


@njit
def sim_meshgrid(shape, spacing, origin, dtype=np.float32):
    """
    https://stackoverflow.com/questions/70613681/numba-compatible-numpy-meshgrid
    """
    out_length = 1
    for i in range(len(shape)):
        out_length *= shape[i]

    coords = np.empty(
        (
            out_length,
            len(shape),
        ),
        dtype=dtype,
    )

    for i, index in enumerate(np.ndindex(shape)):
        for idim in range(len(shape)):
            coords[i, idim] = index[idim] * spacing[idim] + origin[idim]

    return coords
