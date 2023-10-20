import numpy as np
import xarray as xr
from scipy.spatial.transform import Rotation


def affine_from_translation(translation):
    """
    Return matrix in homogeneous coords representing a translation.
    """
    ndim = len(translation)
    M = np.concatenate([translation, [1]], axis=0)
    M = np.concatenate([np.eye(ndim + 1)[:, :ndim], M[:, None]], axis=1)
    return M


def affine_from_rotation(angle, direction, point=None):
    """
    Return matrix in homogeneous coords to rotate around axis
    defined by point and direction.
    """

    R = Rotation.from_rotvec(angle * np.array(direction)).as_matrix()

    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)

    return M


def identity_transform(ndim, t_coords=None):
    if t_coords is None:
        params = xr.DataArray(np.eye(ndim + 1), dims=["x_in", "x_out"])
    else:
        params = xr.DataArray(
            len(t_coords) * [np.eye(ndim + 1)],
            dims=["t", "x_in", "x_out"],
            coords={"t": t_coords},
        )

    return params


# def affine_to_xaffine(affine, t_coords=None):

#     if t_coords is None:
#         params = xr.DataArray(affine, dims=["x_in", "x_out"])
#     else:
#         params = xr.DataArray(
#             len(t_coords) * [affine],
#             dims=["t", "x_in", "x_out"],
#             coords={"t": t_coords},
#         )

#     return params


def matmul_xparams(xparams1, xparams2):
    return xr.apply_ufunc(
        np.matmul,
        xparams1,
        xparams2,
        input_core_dims=[["x_in", "x_out"]] * 2,
        output_core_dims=[["x_in", "x_out"]],
        dask="parallelized",
        vectorize=True,
        join="inner",
    )


def invert_xparams(xparams):
    return xr.apply_ufunc(
        np.linalg.inv,
        xparams,
        input_core_dims=[["x_in", "x_out"]],
        output_core_dims=[["x_in", "x_out"]],
        vectorize=False,
        # dask='allowed',
        dask="parallelized",
    )
