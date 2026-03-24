from typing import Union

import numpy as np
import xarray as xr
from scipy.fft import dctn
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.optimize import minimize
from spatial_image import to_spatial_image

from multiview_stitcher import transformation

BoundingBox = dict[str, dict[str, Union[float, int]]]

try:
    import cupy as cp
    import cupyx.scipy.ndimage
except ImportError:
    cp = None


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
    elif method_func == dct_shannon_entropy:
        return 0  # DCT method doesn't require specific overlap
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
        nan_gaussian_filter(
            (
                sim_t
                - nan_gaussian_filter(
                    sim_t,
                    sigma=sigma_1,
                    mode="reflect"
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


def dct_shannon_entropy(
    transformed_views,
    blending_weights,
    axes=None,
    how_many_best_views=2,
    cumulative_weight_best_views=0.9,
):
    """
    Calculate weights using DCT Shannon Entropy.

    Based on the method described in:
    "Adaptive light-sheet microscopy for long-term, high-resolution imaging in living organisms"
    http://www.nature.com/articles/nbt.3708

    This method computes the Discrete Cosine Transform (DCT) of each view and calculates
    an entropy-based quality metric. It then applies a heuristic to adapt weights to the
    number of views, polarizing them so that a small number of best views contribute
    most of the weight.

    Parameters
    ----------
    transformed_views : array-like of shape (n_views, z, y, x) or (n_views, y, x)
        Input images containing only spatial dimensions.
    blending_weights : array-like
        Blending weights for each view (used to mask invalid regions).
    axes : list of int, optional
        Axes along which to compute DCT. If None, uses all spatial axes.
    how_many_best_views : int, optional
        Number of best views that should carry most of the weight. Default is 2.
    cumulative_weight_best_views : float, optional
        Target cumulative weight for the best views (e.g., 0.9 means best views
        should contribute 90% of total weight). Default is 0.9.

    Returns
    -------
    weights : np.ndarray of shape (n_views, z, y, x) or (n_views, y, x)
        Entropy-based weights for each view.
    """

    vrs = np.copy(transformed_views)
    
    # Set default axes (all spatial dimensions)
    if axes is None:
        axes = list(range(1, vrs.ndim))
    
    ds = []
    for v in vrs:
        # Skip views that are mostly zero
        if np.sum(v == 0) > np.prod(v.shape) * (4 / 5.):
            ds.append([0])
            continue
        elif v.min() < 0.0001:
            # Replace zeros with minimum positive value
            v[v == 0] = v[v > 0].min()
        
        # Compute DCT
        d = dctn(v, norm='ortho', axes=[ax - 1 for ax in axes])
        ds.append(d.flatten())
    
    # L2 norm
    dsl2 = np.array([np.sum(np.abs(d)) for d in ds])
    # Don't divide by zero
    dsl2[dsl2 == 0] = 1
    
    def abslog(x):
        """Compute absolute value of log2, handling zeros."""
        res = np.zeros_like(x)
        x = np.abs(x)
        res[x == 0] = 0
        res[x > 0] = np.log2(x[x > 0])
        return res
    
    # Calculate entropy-based weights
    ws = np.array([-np.sum(np.abs(d) * abslog(d / dsl2[id])) for id, d in enumerate(ds)])
    
    # Simple uniform weights if everything is zero
    if not ws.max():
        ws = np.ones(len(ws)) / float(len(ws))
    
    # HEURISTIC to adapt weights to number of views
    # Polarize weights so that the best few views carry most of the weight
    if len(ws) > 2 and ws.min() < ws.max():
        # Normalize weights
        wsum = np.sum(ws, 0)
        for iw, w in enumerate(ws):
            ws[iw] /= wsum
        
        wf = ws
        wfs = np.sort(wf, axis=0)
        
        def energy(exp):
            """Objective function to find optimal exponent."""
            exp = exp[0]
            tmpw = wfs ** exp
            tmpsum = np.sum(tmpw, 0)
            tmpw = tmpw / tmpsum
            
            # Sum of weights for the best N views
            nsum = np.sum(tmpw[-int(how_many_best_views):], (-1))
            energy = np.abs(np.sum(nsum) - cumulative_weight_best_views)
            
            return energy
        
        # Find optimal exponent to polarize weights
        res = minimize(
            energy, 
            [0.5], 
            bounds=[[0.1, 10]], 
            method='L-BFGS-B', 
            options={'maxiter': 10}
        )
        
        exp = res.x[0]
        
        # Apply exponent to polarize weights
        ws = [ws[i] ** exp for i in range(len(ws))]
        ws = np.array(ws)
    
    # Reshape to match input dimensions
    reshape_dims = [ws.shape[0]] + [1] * (transformed_views.ndim - 1)
    ws = ws.reshape(reshape_dims)
    
    # Normalize final weights
    ws_expanded = np.broadcast_to(ws, transformed_views.shape)
    ws_expanded = normalize_weights(ws_expanded)
    
    return ws_expanded


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


def get_blending_weights(
    target_bb: BoundingBox,
    source_bb: BoundingBox,
    affine: xr.DataArray,
    blending_widths: dict[str, float] = None,
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
    params : list of xarray.DataArray
        Transformation parameters for each view.
    blending_widths : dict
        Physical blending widths for each dimension.

    Returns
    -------
    target_weights : dask array containing blending weights
        for the target bounding box
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
