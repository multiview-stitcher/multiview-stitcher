from typing import Union

import numpy as np
import xarray as xr

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

    cuboid_coords = np.zeros((len(pts), ndim))
    for idim, _dim in enumerate(sdims):
        cuboid_coords[:, idim] = np.dot(pts - ot, cuboid_vectors_t[idim])

    # vectorize further
    cuboid_coords = np.dot(pts - ot, cuboid_vectors_t.T)

    # check if pt is in cuboid
    insides = np.min(
        [cuboid_coords > 0.0, cuboid_coords <= extent], axis=(0, -1)
    )

    # distance from pt to cuboid
    distances = np.min([cuboid_coords, extent - cuboid_coords], axis=0)
    if dim_factors is not None:
        distances = distances * np.array([dim_factors[dim] for dim in sdims])

    distances = np.min(distances, axis=-1)

    # set distances to 0 for points outside
    distances = distances * insides

    return distances
