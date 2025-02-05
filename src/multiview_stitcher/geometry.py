from typing import Union

import numpy as np
import xarray as xr
from numba import njit

from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher import transformation

BoundingBox = dict[str, dict[str, Union[float, int]]]


def interior_distance(
    pts: np.ndarray,
    sp: BoundingBox,
    affine: list[xr.DataArray],
    dim_factors: dict[str, Union[float, int]],
):
    """
    Get the distance of points to the border of a cuboid defined by the stack properties sp.
    Distances are positive for points inside the cuboid and zero for points outside.

    Parameters
    ----------
    pts : np.ndarray
        Points to calculate distance for (in world coordinates).
    sp : BoundingBox
        Bounding box of the cuboid.
    affine : list[xr.DataArray]
        Affine transformations mapping sp into world coordinates.
    dim_factors : dict
        Factors to scale distances in each dimension.

    Notes
    -----
    - define cuboid coordinate system
    - convert point to cuboid coordinate system
    - check if positive and smaller than 1 / shape extent
    """

    sdims = si_utils.get_spatial_dims_from_stack_properties(sp)
    ndim = len(sdims)

    # origin of cuboid
    # Note: origin is at the center of the first voxel, not the origin of the cuboid
    s = np.array([sp["spacing"][dim] for dim in sdims])
    n = np.array([sp["shape"][dim] for dim in sdims])
    o = np.array(
        [sp["origin"][dim] - 0.5 * sp["spacing"][dim] for dim in sdims]
    )

    ot = transformation.transform_pts([o], affine)[0]
    extent = s * n

    cuboid_vectors = np.zeros((ndim, ndim))
    for idim, _dim in enumerate(sdims):
        # unit vector in direction of dimension
        cuboid_vectors[idim, idim] = 1.0

    # get cuboid vectors in world space
    cuboid_vectors_t = np.zeros((ndim, ndim))
    for idim, _dim in enumerate(sdims):
        cuboid_vectors_t[idim] = (
            transformation.transform_pts([o + cuboid_vectors[idim]], affine)[0]
            - ot
        )

    cuboid_vectors_t = cuboid_vectors_t.astype(np.float32)
    ot = ot.astype(np.float32)
    extent = extent.astype(np.float32)

    # project
    cuboid_coords = dot(pts - ot, cuboid_vectors_t.T)

    distances = process_cuboid_coords(
        cuboid_coords,
        extent,
        tuple([dim_factors[dim] for dim in sdims]),
    )

    return distances


@njit
def dot(arrA, arrB):
    """numpy.dot as function takes 2 arguments."""
    return np.dot(arrA, arrB)


@njit
def process_cuboid_coords(cuboid_coords, extent, dim_factors):
    """
    Process cuboid coordinates to get distances.

    Parameters
    ----------
    cuboid_coords : np.ndarray
        Cuboid coordinates.

    Returns
    -------
    tuple
        Tuple of distances and insides.
    """
    distances = np.ones(cuboid_coords.shape[0], dtype=np.float32) * np.inf

    for i, coords in enumerate(cuboid_coords):
        for idim, coord in enumerate(coords):
            if coord < 0.0 or coord > extent[idim]:
                distances[i] = 0
                break
            else:
                distances[i] = min(
                    [
                        distances[i],
                        min([coord, extent[idim] - coord]) * dim_factors[idim],
                    ]
                )

    return distances
