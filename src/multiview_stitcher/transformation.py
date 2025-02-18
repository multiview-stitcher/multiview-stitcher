import dask.array as da
import numpy as np
import spatial_image as si
from dask_image.ndinterp import affine_transform as dask_image_affine_transform
from scipy.ndimage import affine_transform

from multiview_stitcher import param_utils, spatial_image_utils


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

    matrix = p[:ndim, :ndim]
    offset = p[:ndim, ndim]

    # spacing matrices
    Sx = np.diag(
        [output_stack_properties["spacing"][dim] for dim in spatial_dims]
    )
    Sy = np.diag(spatial_image_utils.get_spacing_from_sim(sim, asarray=True))

    Ox = np.array(
        [output_stack_properties["origin"][dim] for dim in spatial_dims]
    )
    Oy = spatial_image_utils.get_origin_from_sim(sim, asarray=True)

    matrix_prime = np.dot(np.linalg.inv(Sy), np.dot(matrix, Sx))
    offset_prime = np.dot(
        np.linalg.inv(Sy),
        offset
        - Oy
        + np.dot(
            matrix,
            Ox,
        ),
    )

    # take care of floating point errors: round parameters to 10 decimals
    # TODO: This is a hack, we should find a better way to deal with floating
    # point errors. Could not reproduce the following error in a simple test yet:
    # In a fractal stitching task: fused output contained empty z slices.

    decimals = 10
    matrix_prime = np.around(matrix_prime, decimals=decimals)
    offset_prime = np.around(offset_prime, decimals=decimals)

    affine_transform_kwargs = {
        "matrix": matrix_prime,
        "offset": offset_prime,
        "output_shape": tuple(
            [output_stack_properties["shape"][dim] for dim in spatial_dims]
        ),
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
    pts = np.array(pts)
    pts = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    pts_t = np.array([np.dot(np.array(affine), pt) for pt in pts])
    pts_t = pts_t[:, :-1]

    return pts_t
