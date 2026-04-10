from typing import Union

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

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
    transformed_views,
    blending_weights,
    sigma_1=5,
    sigma_2=11,
    backend=None,
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
    backend : Backend, optional
        If provided, arrays are converted from backend to numpy
        (gaussian filtering requires scipy on CPU).

    Returns
    -------
    weights : da.array of dim (n_views, (z,) y, x)
        Content based weights for each view.
    """

    # Gaussian filtering requires scipy on CPU
    if backend is not None:
        transformed_views = backend.to_numpy(transformed_views)
        blending_weights = backend.to_numpy(blending_weights)

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
    # Data is on CPU here (to_numpy above), so normalize on the CPU path.
    weights = normalize_weights(weights, backend=None)

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


def normalize_weights(weights, backend=None):
    """
    Normalize weights.

    Parameters
    ----------
    weights : da.array of dim (n_views, (z,) y, x)
        Weights to normalize.
    backend : Backend, optional
        If provided, delegate to backend.normalize_weights().

    Returns
    -------
    weights : da.array of dim (n_views, (z,) y, x)
        Normalized weights.
    """
    if backend is not None:
        return backend.normalize_weights(weights)

    from multiview_stitcher._numba_acceleration import (
        normalize_weights as _accel_normalize,
    )

    return _accel_normalize(weights)


def _blending_weights_edt_and_spacing(source_bb, blending_widths, sdims):
    """
    Compute the EDT mask, support spacing, and origin used for blending.

    Shared between the SpatialImage path and the raw-array backend path.
    """
    ndim = len(sdims)

    mask = np.zeros([3 + 2 for dim in sdims])
    mask[(slice(1, -1),) * ndim] = 1

    support_spacing = {
        dim: (source_bb["shape"][dim] - 1) / 4 * source_bb["spacing"][dim]
        for dim in sdims
    }

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

    return mask, edt_support_spacing, edt_support_origin


def get_blending_weights(
    target_bb: BoundingBox,
    source_bb: BoundingBox,
    affine: xr.DataArray,
    blending_widths: dict[str, float] = None,
    backend=None,
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
    affine : xarray.DataArray
        Transformation parameters.
    blending_widths : dict
        Physical blending widths for each dimension.
    backend : Backend, str, or None
        Compute backend. None uses the global default.

    Returns
    -------
    target_weights : ndarray
        Blending weights as a raw array on the given backend.
    """
    from multiview_stitcher.backends import get_backend

    backend = get_backend(backend)

    if blending_widths is None:
        blending_widths = {"z": 3, "y": 10, "x": 10}

    sdims = sorted(source_bb["origin"].keys())[::-1]

    mask, edt_support_spacing, edt_support_origin = (
        _blending_weights_edt_and_spacing(source_bb, blending_widths, sdims)
    )

    edt_support = backend.distance_transform_edt(
        mask,
        sampling=[
            edt_support_spacing[dim] / blending_widths[dim]
            for dim in sdims
        ],
    )
    edt_support = backend.asarray(edt_support, dtype=np.float32)

    target_weights = transformation.transform_data(
        edt_support,
        p=np.linalg.inv(affine),
        input_spacing=np.array([edt_support_spacing[dim] for dim in sdims]),
        input_origin=np.array([edt_support_origin[dim] for dim in sdims]),
        output_stack_properties=target_bb,
        spatial_dims=sdims,
        backend=backend,
        order=1,
        cval=0.0,
    )

    # Cosine blending via half-angle identity:
    #   (cos((1-w)*pi) + 1) / 2  ==  sin²(w*pi/2)    for w in [0, 1]
    # The sin² form is numerically stable in fp32 (no catastrophic
    # cancellation when w ≈ 0) and works on all devices including TPU.
    # Values >= 1 are already fully inside the tile and kept unchanged.
    below_one = target_weights < 1
    blended = backend.sin(backend.clip(target_weights, 0, 1) * (backend.pi / 2)) ** 2
    target_weights = backend.where(below_one, blended, target_weights)
    del below_one, blended

    return target_weights
