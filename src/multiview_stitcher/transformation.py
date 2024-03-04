import copy
import warnings
from collections.abc import Iterable
from itertools import product

import numpy as np
import spatial_image as si
import xarray as xr
from dask_image.ndinterp import affine_transform as dask_image_affine_transform

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
