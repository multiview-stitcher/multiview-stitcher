import dask.array as da
import numpy as np
import spatial_image as si
from dask_image.ndinterp import affine_transform as dask_image_affine_transform
from multiview_stitcher._numba_acceleration import affine_transform

from multiview_stitcher import param_utils, spatial_image_utils


def _compute_affine_params(
    p,
    input_spacing,
    input_origin,
    output_stack_properties,
    spatial_dims,
):
    """
    Compute the matrix and offset for an affine transform in pixel space.

    Parameters
    ----------
    p : array-like
        Homogeneous transformation matrix (ndim+1, ndim+1).
    input_spacing : array-like
        Spacing of the input image (1D, length ndim).
    input_origin : array-like
        Origin of the input image (1D, length ndim).
    output_stack_properties : dict
        Must contain 'spacing', 'origin', 'shape' keyed by spatial dim names.
    spatial_dims : tuple of str
        Spatial dimension names in order.

    Returns
    -------
    matrix_prime : ndarray
        Affine matrix in pixel coordinates.
    offset_prime : ndarray
        Affine offset in pixel coordinates.
    output_shape : tuple of int
        Shape of the output array.
    """
    ndim = len(spatial_dims)
    matrix = p[:ndim, :ndim]
    offset = p[:ndim, ndim]

    Sx = np.diag(
        [output_stack_properties["spacing"][dim] for dim in spatial_dims]
    )
    Sy = np.diag(np.asarray(input_spacing))
    Ox = np.array(
        [output_stack_properties["origin"][dim] for dim in spatial_dims]
    )
    Oy = np.asarray(input_origin)

    matrix_prime = np.dot(np.linalg.inv(Sy), np.dot(matrix, Sx))
    offset_prime = np.dot(
        np.linalg.inv(Sy),
        offset - Oy + np.dot(matrix, Ox),
    )

    decimals = 10
    matrix_prime = np.around(matrix_prime, decimals=decimals)
    offset_prime = np.around(offset_prime, decimals=decimals)

    output_shape = tuple(
        [output_stack_properties["shape"][dim] for dim in spatial_dims]
    )

    return matrix_prime, offset_prime, output_shape


def transform_data(
    data,
    p,
    input_spacing,
    input_origin,
    output_stack_properties,
    spatial_dims,
    backend=None,
    **affine_transform_kwargs,
):
    """
    Apply an affine transform to raw array data using the given backend.

    This is the low-level workhorse used by both ``transform_sim`` and
    the fusion pipeline. It operates on raw arrays (numpy or backend
    arrays) and returns a raw array on the same backend.

    Parameters
    ----------
    data : array-like
        Input array (numpy or backend array).
    p : array-like
        Homogeneous transformation matrix (ndim+1, ndim+1).
    input_spacing : array-like
        Spacing of the input image.
    input_origin : array-like
        Origin of the input image.
    output_stack_properties : dict
        Must contain 'spacing', 'origin', 'shape' keyed by dim names.
    spatial_dims : tuple of str
        Spatial dimension names in order.
    backend : Backend or None
        Compute backend. If None, uses the global default.
    **affine_transform_kwargs
        Extra kwargs forwarded to the backend's ``affine_transform``
        (e.g. ``mode``, ``cval``, ``order``).

    Returns
    -------
    result : array
        Transformed array on the given backend.
    """
    from multiview_stitcher.backends import get_backend

    backend = get_backend(backend)

    matrix_prime, offset_prime, output_shape = _compute_affine_params(
        p, input_spacing, input_origin, output_stack_properties, spatial_dims,
    )

    kwargs = {
        "matrix": matrix_prime,
        "offset": offset_prime,
        "output_shape": output_shape,
        "mode": "constant",
        "cval": 0.0,
        "order": 1,
    } | affine_transform_kwargs

    return backend.affine_transform(data, **kwargs)


def transform_sim(
    sim,
    p=None,
    output_stack_properties=None,
    keep_transform_keys=False,
    **affine_transform_kwargs,
):
    """
    Transform a spatial image

    TODO: Need to have option to low pass filter
    before significantly reducing spacing, see
    https://computergraphics.stackexchange.com/questions/103/do-you-need-to-use-a-lowpass-filter-before-downsizing-an-image
    """

    ndim = spatial_image_utils.get_ndim_from_sim(sim)
    spatial_dims = spatial_image_utils.get_spatial_dims_from_sim(sim)

    if p is None:
        p = param_utils.identity_transform(ndim)

    matrix_prime, offset_prime, output_shape = _compute_affine_params(
        p,
        input_spacing=spatial_image_utils.get_spacing_from_sim(
            sim, asarray=True
        ),
        input_origin=spatial_image_utils.get_origin_from_sim(
            sim, asarray=True
        ),
        output_stack_properties=output_stack_properties,
        spatial_dims=spatial_dims,
    )

    affine_transform_kwargs = {
        "matrix": matrix_prime,
        "offset": offset_prime,
        "output_shape": output_shape,
        "mode": "constant",
        "cval": 0.0,
        "order": 1,
    } | affine_transform_kwargs

    if isinstance(sim.data, da.core.Array):
        out_data = dask_image_affine_transform(
            sim.data,
            **affine_transform_kwargs,
        )
    else:
        out_data = affine_transform(
            sim.data,
            **affine_transform_kwargs,
        )

    sim = si.to_spatial_image(
        out_data,
        dims=sim.dims,
        scale=output_stack_properties["spacing"],
        translation=output_stack_properties["origin"],
    )

    return sim


def transform_pts(pts, affine):
    """
    pts: (M, N)
    affine: (N+1, N+1)
    """
    pts = np.asarray(pts)
    affine = np.asarray(affine)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    return (affine @ pts_h.T).T[:, :-1]
