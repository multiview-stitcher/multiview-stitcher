import dask.array as da
import numpy as np
from dask_image.ndinterp import affine_transform as dask_image_affine_transform
from scipy.ndimage import affine_transform

from multiview_stitcher import param_utils, spatial_image_utils

try:
    import cupy as cp
    import cupyx.scipy.ndimage
except ImportError:
    cp = None


def transform_sim(
    sim,
    p=None,
    output_stack_properties=None,
    keep_transform_keys=False,
    input_spacing=None,
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
    if input_spacing is None:
        input_spacing = spatial_image_utils.get_spacing_from_sim(sim)
    Sy = np.diag([input_spacing[dim] for dim in spatial_dims])

    Ox = np.array(
        [output_stack_properties["origin"][dim] for dim in spatial_dims]
    )
    Oy = spatial_image_utils.get_origin_from_sim(sim, asarray=True)

    # scipy.ndimage.affine_transform maps output pixel coordinates back into
    # input pixel coordinates. Convert the physical transform into that pixel
    # coordinate system before passing it to the backend-specific resampler.
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

    backend_data = spatial_image_utils._get_backend_data(sim)
    input_shape = spatial_image_utils.get_shape_from_sim(sim, asarray=True)

    # Skip resampling when the requested output grid samples exactly the same
    # input pixels. We still rebuild the spatial image below so the output
    # coordinates follow output_stack_properties, matching the resampled path.
    is_noop = (
        tuple(affine_transform_kwargs["output_shape"]) == tuple(input_shape)
        and np.allclose(
            affine_transform_kwargs["matrix"],
            np.eye(ndim),
            rtol=0,
            atol=1e-10,
        )
        and np.allclose(
            affine_transform_kwargs["offset"],
            0,
            rtol=0,
            atol=1e-10,
        )
    )

    if is_noop:
        out_data = backend_data
    elif spatial_image_utils.is_dask_backed_dataarray(sim):
        out_data = dask_image_affine_transform(
            backend_data,
            **affine_transform_kwargs,
        )
    else:
        # check if sim.data is a cupy array
        if cp is not None and isinstance(backend_data, cp.ndarray):
            # use cupyx.scipy.ndimage for affine transform
            matrix = cp.asarray(affine_transform_kwargs.pop("matrix"))
            out_data = cupyx.scipy.ndimage.affine_transform(
                backend_data,
                matrix=matrix,
                **affine_transform_kwargs,
            )
        else:
            out_data = affine_transform(
                backend_data,
                **affine_transform_kwargs,
            )

    sim = spatial_image_utils.to_spatial_image(
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
