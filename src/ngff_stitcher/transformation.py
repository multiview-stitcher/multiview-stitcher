from collections.abc import Iterable

import numpy as np
import spatial_image as si
from dask_image.ndinterp import affine_transform as dask_image_affine_transform

from ngff_stitcher import spatial_image_utils


def transform_sim(
    sim,
    p=None,
    output_shape=None,
    output_spacing=None,
    output_origin=None,
    output_chunksize=256,
    order=1,
):
    """
    (Lazily) transform a spatial image
    """

    ndim = spatial_image_utils.get_ndim_from_sim(sim)

    if p is None:
        p = np.eye(ndim + 1)

    if output_shape is None:
        output_shape = spatial_image_utils.get_shape_from_sim(
            sim, asarray=True
        )

    if output_spacing is None:
        output_spacing = spatial_image_utils.get_spacing_from_sim(
            sim, asarray=True
        )

    if output_origin is None:
        output_origin = spatial_image_utils.get_origin_from_sim(
            sim, asarray=True
        )

    ndim = spatial_image_utils.get_ndim_from_sim(sim)
    spatial_dims = spatial_image_utils.get_spatial_dims_from_sim(sim)
    matrix = p[:ndim, :ndim]
    offset = p[:ndim, ndim]

    # spacing matrices
    Sx = np.diag(output_spacing)
    Sy = np.diag(spatial_image_utils.get_spacing_from_sim(sim, asarray=True))

    matrix_prime = np.dot(np.linalg.inv(Sy), np.dot(matrix, Sx))
    offset_prime = np.dot(
        np.linalg.inv(Sy),
        offset
        - spatial_image_utils.get_origin_from_sim(sim, asarray=True)
        + np.dot(matrix, output_origin),
    )

    if isinstance(output_chunksize, Iterable):
        output_chunks = output_chunksize
    else:
        output_chunks = tuple([output_chunksize for _ in output_shape])

    out_da = dask_image_affine_transform(
        sim.data,
        matrix=matrix_prime,
        offset=offset_prime,
        order=order,
        output_shape=tuple(output_shape),
        output_chunks=output_chunks,
        mode="constant",
        cval=0.0,
    )

    sim = si.to_spatial_image(
        out_da,
        dims=sim.dims,
        scale={
            dim: output_spacing[idim] for idim, dim in enumerate(spatial_dims)
        },
        translation={
            dim: output_origin[idim] for idim, dim in enumerate(spatial_dims)
        },
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
