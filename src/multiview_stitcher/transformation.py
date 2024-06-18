import copy
import warnings
from collections.abc import Iterable
from itertools import product

import numpy as np
import scipy
import spatial_image as si
import xarray as xr
from dask import array as da
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
from scipy.ndimage import affine_transform as ndimage_affine_transform

from multiview_stitcher import param_utils, spatial_image_utils


def transform_sim(
    sim,
    p=None,
    output_stack_properties=None,
    output_chunksize=None,
    order=1,
    cval=0.0,
    keep_transform_keys=False,
):
    """
    (Lazily) transform a spatial image

    TODO: Need to have option to low pass filter
    before significantly reducing spacing, see
    https://computergraphics.stackexchange.com/questions/103/do-you-need-to-use-a-lowpass-filter-before-downsizing-an-image
    """

    ndim = spatial_image_utils.get_ndim_from_sim(sim)
    sdims = spatial_image_utils.get_spatial_dims_from_sim(sim)
    nsdims = [dim for dim in sim.dims if dim not in sdims]

    if output_chunksize is None:
        default_chunksize = spatial_image_utils.get_default_spatial_chunksizes(
            ndim
        )
        output_chunksize = tuple([default_chunksize[dim] for dim in sdims])

    if p is None:
        p = param_utils.identity_transform(ndim)

    if keep_transform_keys:
        transform_attrs = copy.deepcopy(sim.attrs)

    if len(nsdims) > 0:
        merges = []
        for ns_coords in product(
            *tuple([sim.coords[nsdim] for nsdim in nsdims])
        ):
            nscoord_dict = {
                ndsim: ns_coords[i] for i, ndsim in enumerate(nsdims)
            }

            sim_field = spatial_image_utils.sim_sel_coords(sim, nscoord_dict)

            params_coord_dict = {
                ndsim: ns_coords[i]
                for i, ndsim in enumerate(nsdims)
                if ndsim in p.dims
            }
            p_field = p.sel(params_coord_dict)

            sim_field_t = transform_sim(
                sim_field,
                p=p_field,
                output_stack_properties=output_stack_properties,
                output_chunksize=output_chunksize,
                order=order,
                cval=cval,
                keep_transform_keys=keep_transform_keys,
            )

            sim_field_t = sim_field_t.expand_dims(nsdims)
            sim_field_t = sim_field_t.assign_coords(
                {ns_coord.name: [ns_coord.values] for ns_coord in ns_coords}
            )
            merges.append(sim_field_t)

        if len(merges) > 1:
            # suppress pandas future warning occuring within xarray.concat
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=FutureWarning)

                # if sims are named, combine_by_coord returns a dataset
                sim_t = xr.combine_by_coords(
                    [m.rename(None) for m in merges], combine_attrs="drop"
                )
        else:
            sim_t = sim_field_t

        if keep_transform_keys:
            sim_t.attrs.update(transform_attrs)

        return sim_t

    if output_stack_properties is None:
        output_stack_properties = (
            spatial_image_utils.get_stack_properties_from_sim(sim)
        )

    ndim = spatial_image_utils.get_ndim_from_sim(sim)
    spatial_dims = spatial_image_utils.get_spatial_dims_from_sim(sim)
    matrix = p[:ndim, :ndim]
    offset = p[:ndim, ndim]

    # spacing matrices
    Sx = np.diag(
        [output_stack_properties["spacing"][dim] for dim in spatial_dims]
    )
    Sy = np.diag(spatial_image_utils.get_spacing_from_sim(sim, asarray=True))

    matrix_prime = np.dot(np.linalg.inv(Sy), np.dot(matrix, Sx))
    offset_prime = np.dot(
        np.linalg.inv(Sy),
        offset
        - spatial_image_utils.get_origin_from_sim(sim, asarray=True)
        + np.dot(
            matrix,
            [output_stack_properties["origin"][dim] for dim in spatial_dims],
        ),
    )

    if isinstance(output_chunksize, Iterable):
        output_chunks = output_chunksize
    else:
        output_chunks = tuple([output_chunksize for _ in spatial_dims])

    out_da = dask_image_affine_transform(
        sim.data,
        matrix=matrix_prime,
        offset=offset_prime,
        order=order,
        output_shape=tuple(
            [output_stack_properties["shape"][dim] for dim in spatial_dims]
        ),
        output_chunks=output_chunks,
        mode="constant",
        cval=cval,
    )

    sim = si.to_spatial_image(
        out_da,
        dims=sim.dims,
        scale=output_stack_properties["spacing"],
        translation=output_stack_properties["origin"],
    )

    if keep_transform_keys:
        sim.attrs.update(transform_attrs)

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


def dask_image_affine_transform(
    image,
    matrix,
    offset=0.0,
    output_shape=None,
    order=1,
    output_chunks=None,
    **kwargs,
):
    """
    Copied from dask-image to avoid dependency
    (202406: no pyodide support for dask.dataframe).

    Apply an affine transform using Dask. For every
    output chunk, only the slice containing the relevant part
    of the image is processed. Chunkwise processing is performed
    either using `ndimage.affine_transform` or
    `cupyx.scipy.ndimage.affine_transform`, depending on the input type.

    Notes
    -----
        Differences to `ndimage.affine_transformation`:
        - currently, prefiltering is not supported
          (affecting the output in case of interpolation `order > 1`)
        - default order is 1
        - modes 'reflect', 'mirror' and 'wrap' are not supported

        Arguments equal to `ndimage.affine_transformation`,
        except for `output_chunks`.

    Parameters
    ----------
    image : array_like (Numpy Array, Cupy Array, Dask Array...)
        The image array.
    matrix : array (ndim,), (ndim, ndim), (ndim, ndim+1) or (ndim+1, ndim+1)
        Transformation matrix.
    offset : float or sequence, optional
        The offset into the array where the transform is applied. If a float,
        `offset` is the same for each axis. If a sequence, `offset` should
        contain one value for each axis.
    output_shape : tuple of ints, optional
        The shape of the array to be returned.
    order : int, optional
        The order of the spline interpolation. Note that for order>1
        scipy's affine_transform applies prefiltering, which is not
        yet supported and skipped in this implementation.
    output_chunks : tuple of ints, optional
        The shape of the chunks of the output Dask Array.

    Returns
    -------
    affine_transform : Dask Array
        A dask array representing the transformed output

    """

    if not isinstance(image, da.core.Array):
        image = da.from_array(image)

    if output_shape is None:
        output_shape = image.shape

    if output_chunks is None:
        output_chunks = image.shape

    # Perform test run to ensure parameter validity.
    ndimage_affine_transform(np.zeros([0] * image.ndim), matrix, offset)

    # Make sure parameters contained in matrix and offset
    # are not overlapping, i.e. that the offset is valid as
    # it needs to be modified for each chunk.
    # Further parameter checks are performed directly by
    # `ndimage.affine_transform`.

    matrix = np.asarray(matrix)
    offset = np.asarray(offset).squeeze()

    # these lines were copied and adapted from `ndimage.affine_transform`
    if (
        matrix.ndim == 2
        and matrix.shape[1] == image.ndim + 1
        and (matrix.shape[0] in [image.ndim, image.ndim + 1])
    ):
        # assume input is homogeneous coordinate transformation matrix
        offset = matrix[: image.ndim, image.ndim]
        matrix = matrix[: image.ndim, : image.ndim]

    cval = kwargs.pop("cval", 0)
    mode = kwargs.pop("mode", "constant")
    kwargs.pop("prefilter", False)

    supported_modes = ["constant", "nearest"]
    if scipy.__version__ > np.lib.NumpyVersion("1.6.0"):
        supported_modes += ["grid-constant"]
    if mode in ["wrap", "reflect", "mirror", "grid-mirror", "grid-wrap"]:
        raise NotImplementedError(
            f"Mode {mode} is not currently supported. It must be one of "
            f"{supported_modes}."
        )

    n = image.ndim
    image_shape = image.shape

    # calculate output array properties
    normalized_chunks = da.core.normalize_chunks(
        output_chunks, tuple(output_shape)
    )
    block_indices = product(*(range(len(bds)) for bds in normalized_chunks))
    block_offsets = [np.cumsum((0,) + bds[:-1]) for bds in normalized_chunks]

    # use dispatching mechanism to determine backend
    affine_transform_method = ndimage_affine_transform
    asarray_method = np.asarray

    # construct dask graph for output array
    # using unique and deterministic identifier
    output_name = "affine_transform-" + tokenize(
        image, matrix, offset, output_shape, output_chunks, kwargs
    )
    output_layer = {}
    rel_images = []
    for _ib, block_ind in enumerate(block_indices):
        out_chunk_shape = [
            normalized_chunks[dim][block_ind[dim]] for dim in range(n)
        ]
        out_chunk_offset = [
            block_offsets[dim][block_ind[dim]] for dim in range(n)
        ]

        out_chunk_edges = np.array(
            list(np.ndindex(tuple([2] * n)))
        ) * np.array(out_chunk_shape) + np.array(out_chunk_offset)

        # map output chunk edges onto input image coordinates
        # to define the input region relevant for the current chunk
        if matrix.ndim == 1 and len(matrix) == image.ndim:
            rel_image_edges = matrix * out_chunk_edges + offset
        else:
            rel_image_edges = np.dot(matrix, out_chunk_edges.T).T + offset

        rel_image_i = np.min(rel_image_edges, 0)
        rel_image_f = np.max(rel_image_edges, 0)

        # Calculate edge coordinates required for the footprint of the
        # spline kernel according to
        # https://github.com/scipy/scipy/blob/9c0d08d7d11fc33311a96d2ac3ad73c8f6e3df00/scipy/ndimage/src/ni_interpolation.c#L412-L419 # noqa: E501
        # Also see this discussion:
        # https://github.com/dask/dask-image/issues/24#issuecomment-706165593 # noqa: E501
        for dim in range(n):
            if order % 2 == 0:
                rel_image_i[dim] += 0.5
                rel_image_f[dim] += 0.5

            rel_image_i[dim] = np.floor(rel_image_i[dim]) - order // 2
            rel_image_f[dim] = np.floor(rel_image_f[dim]) - order // 2 + order

            if order == 0:  # required for consistency with scipy.ndimage
                rel_image_i[dim] -= 1

        # clip image coordinates to image extent
        for dim, s in zip(range(n), image_shape):
            rel_image_i[dim] = np.clip(rel_image_i[dim], 0, s - 1)
            rel_image_f[dim] = np.clip(rel_image_f[dim], 0, s - 1)

        rel_image_slice = tuple(
            [
                slice(int(rel_image_i[dim]), int(rel_image_f[dim]) + 2)
                for dim in range(n)
            ]
        )

        rel_image = image[rel_image_slice]

        """Block comment for future developers explaining how `offset` is
        transformed into `offset_prime` for each output chunk.
        Modify offset to point into cropped image.
        y = Mx + o
        Coordinate substitution:
        y' = y - y0(min_coord_px)
        x' = x - x0(chunk_offset)
        Then:
        y' = Mx' + o + Mx0 - y0
        M' = M
        o' = o + Mx0 - y0
        """

        offset_prime = offset + np.dot(matrix, out_chunk_offset) - rel_image_i

        output_layer[(output_name,) + block_ind] = (
            affine_transform_method,
            (da.core.concatenate3, rel_image.__dask_keys__()),
            asarray_method(matrix),
            offset_prime,
            tuple(out_chunk_shape),  # output_shape
            None,  # out
            order,
            mode,
            cval,
            False,  # prefilter
        )

        rel_images.append(rel_image)

    graph = HighLevelGraph.from_collections(
        output_name, output_layer, dependencies=[image] + rel_images
    )

    meta = asarray_method([0]).astype(image.dtype)

    transformed = da.Array(
        graph,
        output_name,
        shape=tuple(output_shape),
        # chunks=output_chunks,
        chunks=normalized_chunks,
        meta=meta,
    )

    return transformed
