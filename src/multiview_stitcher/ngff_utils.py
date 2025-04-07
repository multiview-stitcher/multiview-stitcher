import os

import ngff_zarr
import numpy as np
import spatial_image as si
import zarr
from dask import array as da
from ome_zarr import writer
from ome_zarr.io import parse_url
from xarray import DataTree

from multiview_stitcher import msi_utils, param_utils
from multiview_stitcher import spatial_image_utils as si_utils


def sim_to_ngff_image(sim, transform_key):
    """
    Convert a spatial_image (multiview-stitcher flavor) into a
    ngff_image in-memory representation compatible with NGFF v0.4.

    The translational component of the affine transform associated to
    the given transform_key will be added to the
    `translate` coordinateTransformation of the NGFF image.
    """

    sdims = si_utils.get_spatial_dims_from_sim(sim)
    nsdims = si_utils.get_nonspatial_dims_from_sim(sim)

    origin = si_utils.get_origin_from_sim(sim)
    if transform_key is not None:
        transform = si_utils.get_affine_from_sim(sim, transform_key)
        for nsdim in nsdims:
            if nsdim in transform.dims:
                transform = transform.sel(
                    {
                        nsdim: transform.coords[nsdim][0]
                        for nsdim in transform.dims
                    }
                )
        transform = np.array(transform)
        transform_translation = param_utils.translation_from_affine(transform)
        for isdim, sdim in enumerate(sdims):
            origin[sdim] = origin[sdim] + transform_translation[isdim]

    ngff_im = ngff_zarr.to_ngff_image(
        sim.data,
        dims=sim.dims,
        scale=si_utils.get_spacing_from_sim(sim),
        translation=origin,
    )

    return ngff_im


def msim_to_ngff_multiscales(msim, transform_key):
    """
    Convert a multiscale_spatial_image (multiview-stitcher flavor) into a
    ngff_image in-memory representation compatible with NGFF v0.4.

    The translational component of the affine transform associated to
    the given transform_key will be added to the
    `translate` coordinateTransformation of the NGFF image(s).
    """

    ngff_ims = []
    for scale_key in msi_utils.get_sorted_scale_keys(msim):
        sim = msi_utils.get_sim_from_msim(msim, scale=scale_key)
        ngff_ims.append(sim_to_ngff_image(sim, transform_key=transform_key))

    # workaround for creating multiscale metadata
    # does this create significant overhead?
    ngff_multiscales_scales = [
        ngff_zarr.to_multiscales(ngff_im, scale_factors=[])
        for ngff_im in ngff_ims
    ]

    sdims = msi_utils.get_spatial_dims(msim)

    ngff_multiscales = ngff_zarr.Multiscales(
        ngff_ims,
        metadata=ngff_zarr.Metadata(
            axes=ngff_multiscales_scales[0].metadata.axes,
            datasets=[
                ngff_zarr.Dataset(
                    path="scale%s/image" % iscale,
                    coordinateTransformations=ngff_multiscales_scale.metadata.datasets[
                        0
                    ].coordinateTransformations,
                )
                for iscale, ngff_multiscales_scale in enumerate(
                    ngff_multiscales_scales
                )
            ],
            coordinateTransformations=None,
        ),
        scale_factors=[
            {
                sdim: int(
                    ngff_ims[0].data.shape[ngff_ims[0].dims.index(sdim)]
                    / ngff_ims[iscale].data.shape[
                        ngff_ims[iscale].dims.index(sdim)
                    ]
                )
                for sdim in sdims
            }
            for iscale in range(1, len(ngff_ims))
        ],
    )

    return ngff_multiscales


def ngff_image_to_sim(ngff_im, transform_key):
    """
    Convert a ngff_image in-memory representation compatible with NGFF v0.4
    into a spatial_image (multiview-stitcher flavor).
    """

    sim = si.to_spatial_image(
        ngff_im.data,
        dims=ngff_im.dims,
        scale=ngff_im.scale,
        translation=ngff_im.translation,
    )

    sim = si_utils.ensure_dim(sim, "t")

    if "c" not in sim.dims:
        sim = sim.expand_dims(["c"])

    sdims = si_utils.get_spatial_dims_from_sim(sim)

    si_utils.set_sim_affine(
        sim,
        param_utils.affine_to_xaffine(
            np.eye(len(sdims) + 1), t_coords=sim.coords["t"].values
        ),
        transform_key=transform_key,
    )

    return sim


def ngff_multiscales_to_msim(ngff_multiscales, transform_key):
    """
    Convert a list of ngff_image in-memory representations compatible with NGFF v0.4
    into a multiscale_spatial_image (multiview-stitcher flavor).
    """

    msim_dict = {}
    for iscale, ngff_im in enumerate(ngff_multiscales.images):
        sim = ngff_image_to_sim(ngff_im, transform_key=transform_key)
        curr_scale_msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
        msim_dict[f"scale{iscale}"] = curr_scale_msim["scale0"]

    msim = DataTree.from_dict(msim_dict)

    return msim


def write_sim_to_ome_zarr(
    sim,
    output_zarr_url,
    downscale_factors_per_spatial_dim=None,
    overwrite=False,
):
    """
    Write (and compute) a spatial_image (multiview-stitcher flavor)
    to a multiscale NGFF zarr file (v0.4).
    Returns a sim backed by the newly created zarr file.

    If overwrite is False, image data will be read from the zarr file
    and missing pyramid levels will be completed. OME-Zarr metadata
    will be overwritten in any case.

    Note that any transform_key will not be stored in the zarr file.
    However, the returned sim will have the transform_key set.
    """

    # if not overwrite and os.path.exists(f"{output_zarr_url}/0"):
    #     # warn that the file already exists

    #     warnings.warn(
    #         f"File {output_zarr_url}/0 already exists. "
    #         "Use overwrite=True to overwrite it.",
    #         UserWarning,
    #         stacklevel=1,
    #     )

    #     sim.data = da.from_zarr(f"{output_zarr_url}/0")
    #     return sim

    ndim = sim.data.ndim
    dims = sim.dims
    sdims = si_utils.get_spatial_dims_from_sim(sim)
    spacing = si_utils.get_spacing_from_sim(sim)
    origin = si_utils.get_origin_from_sim(sim)
    spatial_shape = {
        dim: sim.data.shape[idim]
        for idim, dim in enumerate(dims)
        if dim in sdims
    }

    if downscale_factors_per_spatial_dim is None:
        downscale_factors_per_spatial_dim = {dim: 2 for dim in sdims}

    res_shapes = [spatial_shape]
    res_rel_factors = [{dim: 1 for dim in sdims}]
    res_abs_factors = [{dim: 1 for dim in sdims}]
    while True:
        new_rel_factors = {
            dim: downscale_factors_per_spatial_dim[dim]
            if res_shapes[-1][dim] // downscale_factors_per_spatial_dim[dim]
            > 10
            else 1
            for dim in sdims
        }

        new_abs_factors = {
            dim: res_abs_factors[-1][dim] * new_rel_factors[dim]
            for dim in sdims
        }
        new_shape = {
            dim: res_shapes[-1][dim] // new_rel_factors[dim] for dim in sdims
        }
        if not any(new_rel_factors[dim] > 1 for dim in sdims):
            break

        res_shapes.append(new_shape)
        res_rel_factors.append(
            new_rel_factors | {dim: 1 for dim in dims if dim not in sdims}
        )
        res_abs_factors.append(
            new_abs_factors | {dim: 1 for dim in dims if dim not in sdims}
        )

    n_resolutions = len(res_shapes)

    if not overwrite and os.path.exists(f"{output_zarr_url}/0"):
        sim.data = da.from_zarr(
            f"{output_zarr_url}/0",
            dimension_separator="/",
        )
        top_level_exists = True
    else:
        # Open output array. This allows setting `write_empty_chunks=True`,
        # which cannot be passed to dask.array.to_zarr below.
        output_zarr_arr = zarr.open(
            f"{output_zarr_url}/0",
            shape=sim.data.shape,
            chunks=sim.data.chunksize,
            dtype=sim.data.dtype,
            write_empty_chunks=False,
            dimension_separator="/",
            fill_value=0,
            mode="w",
        )

        # Write the lowest resolution
        sim.data = sim.data.to_zarr(
            output_zarr_arr,
            overwrite=True,
            dimension_separator="/",
            return_stored=True,
            compute=True,
        )
        top_level_exists = False

    coordtfs = [
        [
            {
                "type": "scale",
                "scale": [1.0] * (ndim - len(sdims))
                + [
                    float(s * res_abs_factors[res_level][dim])
                    for dim, s in spacing.items()
                ],
            },
            {
                "type": "translation",
                "translation": [0] * (ndim - len(sdims))
                + [
                    origin[dim]
                    + (res_abs_factors[res_level][dim] - 1) * spacing[dim] / 2
                    for dim in sdims
                ],
            },
        ]
        # [0] * (ndim - len(sdims)) + [origin[dim] for dim in sdims]}]
        for res_level in range(n_resolutions)
    ]
    axes = [
        {
            "name": dim,
            "type": "channel"
            if dim == "c"
            else ("time" if dim == "t" else "space"),
        }
        | ({"unit": "micrometer"} if dim in sdims else {})
        for dim in sim.dims
        if dim in dims
    ]

    # thanks to https://github.com/CamachoDejay/teaching-bioimage-analysis-python/blob/6076e00e392075ba9c07e67e868a39d4889e6298/short_examples/zarr-from-tiles/zarr-minimal-example-tiles.ipynb
    def mean_dtype(arr, **kwargs):
        return np.mean(arr, **kwargs).astype(arr.dtype)

    parent_res_array = sim.data
    curr_res_array = sim.data  # in case of only one resolution level
    for res_level in range(1, n_resolutions):
        if not overwrite and os.path.exists(f"{output_zarr_url}/{res_level}"):
            curr_res_array = da.from_zarr(
                f"{output_zarr_url}/{res_level}",
                dimension_separator="/",
            )
        else:
            curr_res_array = da.coarsen(
                mean_dtype,
                parent_res_array,
                axes={
                    idim: res_rel_factors[res_level][dim]
                    for idim, dim in enumerate(dims)
                },
                trim_excess=True,
            )

            curr_res_array = curr_res_array.rechunk(parent_res_array.chunksize)

            # Open output array. This allows setting `write_empty_chunks=True`,
            # which cannot be passed to dask.array.to_zarr below.
            res_level_zarr_arr = zarr.open(
                f"{output_zarr_url}/{res_level}",
                shape=curr_res_array.shape,
                chunks=curr_res_array.chunksize,
                dtype=curr_res_array.dtype,
                write_empty_chunks=False,
                dimension_separator="/",
                fill_value=0,
                mode="w",
            )

            curr_res_array = curr_res_array.to_zarr(
                res_level_zarr_arr,
                overwrite=True,
                dimension_separator="/",
                return_stored=True,
                compute=True,
            )

        parent_res_array = curr_res_array

    if not top_level_exists or overwrite:
        store = parse_url(output_zarr_url, mode="w").store
        output_group = zarr.group(store=store)
        writer.write_multiscales_metadata(
            group=output_group,
            axes=axes,
            datasets=[
                {
                    "path": f"{res_level}",
                    "coordinateTransformations": coordtfs[res_level],
                }
                for res_level in range(n_resolutions)
            ],
        )

        if "c" in sim.dims:
            contrast_min = np.array(
                curr_res_array.min(
                    axis=[
                        idim for idim, dim in enumerate(sim.dims) if dim != "c"
                    ]
                )
            )
            contrast_max = np.array(
                curr_res_array.max(
                    axis=[
                        idim for idim, dim in enumerate(sim.dims) if dim != "c"
                    ]
                )
            )

            output_group.attrs["omero"] = {
                "channels": [
                    {
                        "color": "ffffff",
                        "label": f"{ch}",
                        "active": True,
                        "window": {
                            "end": int(contrast_max[ich]),
                            "max": int(contrast_max[ich]),
                            "min": 0,
                            "start": int(contrast_min[ich]),
                        },
                    }
                    for ich, ch in enumerate(sim.coords["c"].values)
                ],
            }

    return sim


def read_sim_from_ome_zarr(
    zarr_path,
    resolution_level=0,
    transform_key=si_utils.DEFAULT_TRANSFORM_KEY,
):
    """
    Read a multiscale NGFF zarr file (v0.4) into a spatial_image
    (multiview-stitcher flavor) at a given resolution level.

    NGFF zarr files v0.4 cannot contain affine transformations, so
    an identity transform will be set for the given transform_key.

    Parameters
    ----------
    zarr_path : str or Path
        Path to the zarr file
    resolution_level : int, optional
        Resolution level to read, by default 0 (highest resolution)
    transform_key : str, optional
        By default si_utils.DEFAULT_TRANSFORM_KEY

    Returns
    -------
    spatial_image with transform_key set
    """
    ngff_multiscales = ngff_zarr.from_ngff_zarr(zarr_path)

    if resolution_level >= len(ngff_multiscales.images):
        raise ValueError(
            f"Resolution level {resolution_level} not found in {zarr_path}"
        )

    sim = ngff_image_to_sim(
        ngff_multiscales.images[resolution_level], transform_key=transform_key
    )

    # get channel names from omero metadata if available
    store = parse_url(zarr_path, mode="r").store
    root = zarr.group(store=store)

    if "omero" in root.attrs:
        omero = root.attrs["omero"]
        ch_coords = [ch["label"] for ch in omero["channels"]]
        sim = sim.assign_coords(c=ch_coords)

    return sim
