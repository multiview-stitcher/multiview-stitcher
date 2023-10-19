import itertools
import warnings

import dask.array as da
import numpy as np
import xarray as xr

from multiview_stitcher import spatial_image_utils, transformation


def combine_stack_props(stack_props_list):
    combined_stack_props = {}
    combined_stack_props["origin"] = np.min(
        [sp["origin"] for sp in stack_props_list], axis=0
    )
    combined_stack_props["spacing"] = np.min(
        [sp["spacing"] for sp in stack_props_list], axis=0
    )
    combined_stack_props["shape"] = np.max(
        [
            np.ceil(
                (
                    sp["origin"]
                    + sp["shape"] * sp["spacing"]
                    - combined_stack_props["origin"]
                )
                / combined_stack_props["spacing"]
            )
            for sp in stack_props_list
        ],
        axis=0,
    ).astype(np.uint16)

    return combined_stack_props


def fusion_method_weighted_average(
    transformed_views,
    blending_weights,
    fusion_weights=None,
    params=None,
):
    """
    Simple weighted average fusion.
    """

    if fusion_weights is None:
        product = transformed_views * blending_weights
    else:
        product = transformed_views * blending_weights * fusion_weights

    return np.nansum(product, axis=0).astype(transformed_views.dtype)


def fuse(
    sims: list,
    transform_key: str = None,
    fusion_method=fusion_method_weighted_average,
    weights_method=None,
    output_spacing=None,
    output_stack_mode="union",
    output_origin=None,
    output_shape=None,
    output_chunksize=512,
    interpolate_missing_pixels=None,
    # tmpdir=None,
    # multiscale_output=True,
):
    """
    Fuse all fields from CT(Z)YX views
    """

    sdims = spatial_image_utils.get_spatial_dims_from_sim(sims[0])
    nsdims = [dim for dim in sims[0].dims if dim not in sdims]

    if interpolate_missing_pixels is None:
        interpolate_missing_pixels = sdims == 2

    params = [
        spatial_image_utils.get_affine_from_sim(
            sim, transform_key=transform_key
        )
        for sim in sims
    ]

    params = [spatial_image_utils.invert_xparams(param) for param in params]

    if output_spacing is None:
        output_spacing = spatial_image_utils.get_spacing_from_sim(
            sims[0]
        )  # , asarray=False)

    if output_stack_mode is not None:
        output_stack_properties = calc_stack_properties_from_sims_and_params(
            sims,
            params,
            spacing=np.array([output_spacing[dim] for dim in sdims]),
            mode=output_stack_mode,
        )
    else:
        output_stack_properties = {}

    if output_origin is not None:
        output_stack_properties["origin"] = [
            output_origin[dim] for dim in sdims
        ]

    if output_shape is not None:
        output_stack_properties["shape"] = [output_shape[dim] for dim in sdims]

    xds = xr.Dataset(
        # For python >= 3.9 we can use the union '|' operator to merge to dict
        {**{(view, "sim"): sims[view] for view in range(len(sims))},
        **{(view, "param"): params[view] for view in range(len(sims))}},
    )

    merges = []
    for ns_coords in itertools.product(
        *tuple([xds.coords[nsdim] for nsdim in nsdims])
    ):
        sim_coord_dict = {
            ndsim: ns_coords[i] for i, ndsim in enumerate(nsdims)
        }
        params_coord_dict = {
            ndsim: ns_coords[i]
            for i, ndsim in enumerate(nsdims)
            if ndsim in params[0].dims
        }

        ssims = [sim.sel(sim_coord_dict) for sim in sims]
        sparams = [param.sel(params_coord_dict) for param in params]

        merge = fuse_field(
            ssims,
            sparams,
            fusion_method=fusion_method,
            weights_method=weights_method,
            output_origin=output_stack_properties["origin"],
            output_shape=output_stack_properties["shape"],
            output_spacing=output_stack_properties["spacing"],
            output_chunksize=output_chunksize,
            interpolate_missing_pixels=interpolate_missing_pixels,
        )

        merge = merge.expand_dims(nsdims)
        merge = merge.assign_coords(
            {ns_coord.name: [ns_coord.values] for ns_coord in ns_coords}
        )

        merges.append(merge)

    if len(merges) > 1:
        # if sims are named, combine_by_coord returns a dataset

        # suppress pandas future warning occuring within xarray.concat
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)

            res = xr.combine_by_coords([m.rename(None) for m in merges])
    else:
        res = merge

    # if not tmpdir is None:
    #     res.data = da.from_delayed(delayed(
    #         da.to_zarr)(
    #             res.data,
    #             os.path.join(tmpdir.name, res.data.name+'.zarr'),
    #             return_stored=True,
    #             overwrite=True,
    #             compute=True,
    #         ),
    #         shape=res.data.shape,
    #         dtype=res.data.dtype,
    #     )

    # if not tmpdir is None:
    #     res.data = da.to_zarr(
    #             res.data,
    #             os.path.join(tmpdir.name, res.data.name+'.zarr'),
    #             return_stored=True,
    #             overwrite=True,
    #             compute=False,
    #         )

    res = spatial_image_utils.get_sim_from_xim(res)
    spatial_image_utils.set_sim_affine(
        res,
        spatial_image_utils.identity_transform(len(sdims), res.coords["t"]),
        transform_key,
    )

    # if multiscale_output:
    #     res = msi_utils.get_msim_from_sim(res)

    # if not tmpdir is None:
    #     res.data = da.to_zarr(
    #         res.data,
    #         os.path.join(tmpdir.name, res.data.name+'.zarr'),
    #         return_stored=True,
    #         overwrite=True,
    #         compute=False,
    #         )

    return res


def fuse_field(
    sims,
    params,
    fusion_method=fusion_method_weighted_average,
    weights_method=None,
    output_origin=None,
    output_spacing=None,
    output_shape=None,
    output_chunksize=512,
    interpolate_missing_pixels=True,
):
    """
    fuse tiles from single timepoint and channel
    todo: use _transformations and avoid duplication
    """

    input_dtype = sims[0].dtype
    ndim = spatial_image_utils.get_ndim_from_sim(sims[0])

    field_ims_t = []
    field_ws_t = []
    for isim, sim in enumerate(sims):
        blending_widths = [10] * 2 if ndim == 2 else [3] + [10] * 2

        sim_t = transformation.transform_sim(
            sim,
            params[isim],
            output_chunksize=tuple([output_chunksize for _ in output_shape]),
            output_spacing=output_spacing,
            output_shape=output_shape,
            output_origin=output_origin,
            order=1,
        )

        field_ims_t.append(sim_t.data)

        field_w = xr.zeros_like(sim)
        field_w.data = get_smooth_border_weight_from_shape(
            sim.shape[-ndim:],
            chunks=sim.chunks,
            widths=blending_widths,
        )

        field_w_t = transformation.transform_sim(
            field_w,
            params[isim],
            output_chunksize=tuple([output_chunksize for _ in output_shape]),
            output_spacing=output_spacing,
            output_shape=output_shape,
            output_origin=output_origin,
            order=1,
        )

        field_ws_t.append(field_w_t.data)

    field_ims_t = da.stack(field_ims_t)
    field_ws_t = da.stack(field_ws_t)

    wsum = da.nansum(field_ws_t, axis=0)
    wsum[wsum == 0] = 1

    field_ws_t = field_ws_t / wsum

    # calculate fusion weights
    if weights_method is not None:
        fusion_weights = weights_method(
            field_ims_t,
        )
    else:
        fusion_weights = None

    # perform blockwise fusion
    fused_field = da.map_overlap(
        fusion_method,
        *(
            (field_ims_t, field_ws_t)
            if fusion_weights is None
            else (field_ims_t, field_ws_t, fusion_weights)
        ),
        **({"fusion_weights": None} if fusion_weights is None else {}),
        params=params,
        depth={idim: 0 for idim in range(ndim)},
        drop_axis=0,
        dtype=field_ims_t.dtype,
    )

    if interpolate_missing_pixels:
        # find empty spaces
        empty_mask = da.min(da.isnan(field_ws_t), 0)

        # convert to input dtype
        fused_field = fused_field.astype(input_dtype)

        fused_field = da.map_overlap(
            get_interpolated_image,
            fused_field,
            empty_mask,
            depth=tuple(
                [0]
                + [
                    0 if not idim else np.min([s, output_chunksize]) // 4
                    for idim, s in enumerate(fused_field.shape)
                ]
            ),
            dtype=input_dtype,
        )

    if fused_field.dtype != input_dtype:
        fused_field = fused_field.astype(input_dtype)

    fused_field = xr.DataArray(fused_field, dims=sims[0].dims)

    fused_field = spatial_image_utils.assign_si_coords_from_params(
        fused_field,
        spatial_image_utils.compose_params(output_origin, output_spacing),
    )

    return fused_field


def calc_stack_properties_from_sims_and_params(
    sims,
    params,
    spacing,
    mode="union",
):
    """
    considers time
    """

    views_props = []
    for _, sim in enumerate(sims):
        views_props.append(
            {
                "shape": spatial_image_utils.get_shape_from_sim(
                    sim, asarray=True
                ),
                "spacing": spatial_image_utils.get_spacing_from_sim(
                    sim, asarray=True
                ),
                "origin": spatial_image_utils.get_origin_from_sim(
                    sim, asarray=True
                ),
            }
        )

    params_ds = xr.Dataset(dict(enumerate(params)))

    if "t" in params_ds.dims:
        stack_properties = combine_stack_props(
            [
                calc_stack_properties_from_view_properties_and_params(
                    views_props,
                    [
                        params_ds.sel(t=t).data_vars[ip].data
                        for ip in range(len(params))
                    ],
                    spacing=spacing,
                    mode=mode,
                )
                for t in params_ds.coords["t"]
            ]
        )

    else:
        stack_properties = (
            calc_stack_properties_from_view_properties_and_params(
                views_props,
                [p.data for p in params],
                spacing=spacing,
                mode=mode,
            )
        )

    return stack_properties


def calc_stack_properties_from_view_properties_and_params(
    views_props,
    params,
    spacing,
    mode="union",
):
    """
    view props contains
    - shape
    - spacing
    - origin
    """

    spacing = np.array(spacing).astype(float)
    ndim = len(spacing)

    stack_vertices = np.array(list(np.ndindex(tuple([2] * ndim)))).astype(
        float
    )

    if mode == "sample":
        zero_z_face_vertices = stack_vertices[
            np.where(stack_vertices[:, 0] == 0)
        ]
        zero_z_face_vertices[:, 1] = np.mean(
            zero_z_face_vertices[:, 1]
        )  # take mean in x
        transformed_vertices = get_transformed_stack_vertices(
            zero_z_face_vertices, views_props, params
        )
        volume = (
            np.min(transformed_vertices, 0),
            np.max(transformed_vertices, 0),
        )  # lower, upper

    elif mode == "union":
        transformed_vertices = get_transformed_stack_vertices(
            stack_vertices, views_props, params
        )
        volume = np.min(transformed_vertices, 0), np.max(
            transformed_vertices, 0
        )

    elif mode == "intersection":
        transformed_vertices = get_transformed_stack_vertices(
            stack_vertices, views_props, params
        )
        volume = np.max(np.min(transformed_vertices, 1), 0), np.min(
            np.max(transformed_vertices, 1), 0
        )

    stack_properties = calc_stack_properties_from_volume(volume, spacing)

    return stack_properties


def get_transformed_stack_vertices(
    stack_keypoints, stack_properties_list, params
):
    ndim = len(stack_properties_list[0]["spacing"])
    vertices = np.zeros(
        (len(stack_properties_list) * len(stack_keypoints), ndim)
    )
    for iim, sp in enumerate(stack_properties_list):
        tmp_vertices = stack_keypoints * np.array(sp["shape"]) * np.array(
            sp["spacing"]
        ) + np.array(sp["origin"])
        inv_params = np.linalg.inv(params[iim])
        tmp_vertices_transformed = (
            np.dot(inv_params[:ndim, :ndim], tmp_vertices.T).T
            + inv_params[:ndim, ndim]
        )
        vertices[
            iim * len(stack_keypoints) : (iim + 1) * len(stack_keypoints)
        ] = tmp_vertices_transformed

    return vertices


def calc_stack_properties_from_volume(volume, spacing):
    """
    :param volume: lower and upper edge of final volume (e.g. [edgeLow,edgeHigh] as calculated by calc_final_stack_cube)
    :param spacing: final spacing
    :return: dictionary containing size, origin and spacing of final stack
    """

    origin = volume[0]
    shape = np.ceil((volume[1] - volume[0]) / spacing).astype(np.uint16)

    properties_dict = {}
    properties_dict["shape"] = shape
    properties_dict["spacing"] = spacing
    properties_dict["origin"] = origin

    return properties_dict


def smooth_transition(x, x_offset=0.5, x_stretch=None, k=3):
    """
    Transform the distance from the border to a weight for blending.
    """
    # https://math.stackexchange.com/questions/1832177/sigmoid-function-with-fixed-bounds-and-variable-steepness-partially-solved

    if x_stretch is None:
        x_stretch = x_offset

    xp = x
    xp = xp - x_offset  # w is 0 at offset
    xp = xp / x_stretch / 2.0  # w is +/-0.5 at offset +/- x_stretch

    w = 1 - 1 / (1 + (1 / (xp + 0.5) - 1) ** (-k))

    w[xp <= -0.5] = 0.0
    w[xp >= 0.5] = 1.0

    return w


def get_smooth_border_weight_from_shape(shape, chunks, widths=None):
    """
    Get a weight image for blending that is 0 at the border and 1 at the center.
    Transition widths can be specified for each dimension.

    20230926: this adds too many small dask tasks, need to optimize

    Possible better strategy:

    Assume that only the border chunks of the input files need to be different
    from da.ones.

    OR: multiscale strategy
    - first fuse smallest scale
    - then go on with only those chunks that need to be filled in
    - not clear if it's beneficial, as it's already known which chunks are needed

    OR: Manually construct output array

    [!!] OR: Use interpolation to calculate distance map
    - reduce input to one value per chunk, i.e. chunksize of [1]*ndim
    - use this to define which input chunk is required for fusion
    - also use this to calculate weights at border
      (only where above coarse interpolation step gave < 1)

    """

    ndim = len(shape)

    # get distance to border for each dim

    dim_dists = [
        da.arange(shape[dim], chunks=chunks[dim]).astype(float)
        for dim in range(ndim)
    ]

    dim_dists = [
        da.min(da.abs(da.stack([dd, dd - shape[idim]])), axis=0)
        for idim, dd in enumerate(dim_dists)
    ]

    dim_ws = [
        smooth_transition(
            dim_dists[dim], x_offset=widths[dim], x_stretch=widths[dim]
        )
        for dim in range(ndim)
    ]

    # get product of weights for each dim
    w = da.ones(shape, chunks=chunks, dtype=float)
    for dim in range(len(shape)):
        tmp_dim_w = dim_ws[dim]
        for _ in range(ndim - dim - 1):
            tmp_dim_w = tmp_dim_w[:, None]
        w *= tmp_dim_w

    return w


# from scipy import interpolate
def get_interpolated_image(
    image: np.ndarray,
    mask: np.ndarray,
    method: str = "nearest",
    fill_value: int = 0,
):
    """

    Currently only 2d!

    :param image: a 2D image
    :param mask: a 2D boolean image, True indicates missing values
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """

    # in case of no known pixels
    if mask.min():
        return image

    from scipy import interpolate

    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = interpolate.griddata(
        (known_x, known_y),
        known_v,
        (missing_x, missing_y),
        method=method,
        fill_value=fill_value,
    )

    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values

    return interp_image
