from functools import partial
import os, shutil

import dask
import ngff_zarr
import numpy as np
import spatial_image as si
import zarr
from tqdm import tqdm
from dask import array as da
import dask.diagnostics
from ome_zarr import writer
from xarray import DataTree

from multiview_stitcher import msi_utils, param_utils, misc_utils
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


def update_zarr_array_creation_kwargs_for_ngff_version(
    ngff_version, zarr_array_creation_kwargs):

    if zarr_array_creation_kwargs is None:
        zarr_array_creation_kwargs = {}
    if ngff_version == "0.4":
        zarr_array_creation_kwargs.update({
                "dimension_separator": '/',
        })
        if zarr.__version__ >= "3":
            zarr_array_creation_kwargs.update({
                "zarr_format": 2,
            })
    elif ngff_version == "0.5":
        if zarr.__version__ < "3":
            raise ValueError("zarr>=3 required for ngff_version 0.5")
        zarr_array_creation_kwargs.update({
                "zarr_version" if zarr.__version__ < "3"
                else "zarr_format": 3,
        })
    else:
        raise ValueError(f"ngff_version {ngff_version} not supported")
    return zarr_array_creation_kwargs


# thanks to https://github.com/CamachoDejay/teaching-bioimage-analysis-python/blob/6076e00e392075ba9c07e67e868a39d4889e6298/short_examples/zarr-from-tiles/zarr-minimal-example-tiles.ipynb
def mean_dtype(arr, **kwargs):
    return np.mean(arr, **kwargs).astype(arr.dtype)


def write_and_return_downsampled_sim(
    array,
    dims: list[str],
    output_zarr_array_url: str,
    chunksizes: list[int],
    downscale_factors_per_spatial_dim: dict[str, int] = None,
    overwrite: bool = False,
    zarr_array_creation_kwargs: dict = None,
    res_level: int = 0,
    show_progressbar: bool = True,
    n_batch=1,
    batch_func=None,
    batch_func_kwargs=None,
):

    sdims = [dim for dim in dims if dim in si_utils.SPATIAL_DIMS]

    if not overwrite and os.path.exists(output_zarr_array_url):
        print(f"Found existing resolution level {res_level}...")
        array = da.from_zarr(output_zarr_array_url)
    else:
        print(f"Writing resolution level {res_level}...")
        # use pure dask
        if n_batch is None:
            #downscale
            if downscale_factors_per_spatial_dim is not None\
                and np.max(list(downscale_factors_per_spatial_dim.values())) > 1:
                array = da.coarsen(
                    mean_dtype,
                    array,
                    axes={
                        idim: downscale_factors_per_spatial_dim[dim] if dim in sdims else 1
                        for idim, dim in enumerate(dims)
                    },
                    trim_excess=True,
                )

            # Open output array. This allows setting `write_empty_chunks=True`,
            # which cannot be passed to dask.array.to_zarr below.
            output_zarr_arr = zarr.open(
                output_zarr_array_url,
                shape=array.shape,
                chunks=chunksizes,
                dtype=array.dtype,
                config={'write_empty_chunks': True},
                fill_value=0,
                mode="w",
                **zarr_array_creation_kwargs,
            )

            if show_progressbar:
                with dask.diagnostics.ProgressBar(show_progressbar): 
                    # Write the array
                    array = array.to_zarr(
                        output_zarr_arr,
                        overwrite=True,
                        return_stored=True,
                        compute=True,
                    )

            else:
                # Write the array
                array = array.to_zarr(
                    output_zarr_arr,
                    overwrite=True,
                    return_stored=True,
                    compute=True,
                )
        else:
            # use dask with batching to limit memory usage

            output_shape = [np.floor(s) // (downscale_factors_per_spatial_dim[sdim]
                    if sdim in sdims else 1)
                    for s, sdim in zip(array.shape, dims)]

            write_downsampled_chunk_p = partial(write_downsampled_chunk, 
                input_array=array,
                output_shape=output_shape,
                dims=dims,
                output_zarr_array_url=output_zarr_array_url,
                output_chunksizes=chunksizes,
                downscale_factors_per_spatial_dim=downscale_factors_per_spatial_dim,
                zarr_array_creation_kwargs=zarr_array_creation_kwargs,
            )

            normalized_chunks = normalize_chunks(
                shape=output_shape,
                chunks=chunksizes,
            )

            nblocks = [len(nc) for nc in normalized_chunks]

            for batch in tqdm(
                misc_utils.ndindex_batches(nblocks, n_batch),
                total=int(np.ceil(np.prod(nblocks)/n_batch)))\
            if show_progressbar else\
                misc_utils.ndindex_batches(nblocks, n_batch):
                
                if batch_func is None:
                    for block_id in batch:
                        write_downsampled_chunk_p(block_id)
                else:
                    batch_func(
                        write_downsampled_chunk_p, batch,
                        **(batch_func_kwargs or {}))
                    
            array = da.from_zarr(output_zarr_array_url)
    return array


from dask.array.core import normalize_chunks
def write_downsampled_chunk(
    block_id,
    input_array,
    output_shape,
    output_chunksizes,
    dims,
    output_zarr_array_url,
    downscale_factors_per_spatial_dim,
    zarr_array_creation_kwargs,
):

    sdims = [dim for dim in dims if dim in si_utils.SPATIAL_DIMS]
    nsdims = [dim for dim in dims if dim not in si_utils.SPATIAL_DIMS]

    normalized_chunks = normalize_chunks(
        shape=output_shape,
        chunks=output_chunksizes,
    )

    ns_coord = {dim: block_id[idim] for idim, dim in enumerate(nsdims)}
    spatial_chunk_ind = block_id[len(nsdims):]

    chunk_offset = {
        sdims[idim]: int(np.sum(normalized_chunks[len(nsdims) + idim][:b]))
        if b > 0 else 0 for idim, b in enumerate(spatial_chunk_ind)}
    chunk_shape = {
        sdims[idim]: normalized_chunks[len(nsdims) + idim][b]
            for idim, b in enumerate(spatial_chunk_ind)}
    
    input_slices = tuple(
        slice(
            ns_coord[dim],
            ns_coord[dim] + 1,
        )
        if dim in nsdims
        else slice(
            chunk_offset[dim] * (downscale_factors_per_spatial_dim[dim]
                if dim in downscale_factors_per_spatial_dim else 1),
            (chunk_offset[dim] + chunk_shape[dim])
                * (downscale_factors_per_spatial_dim[dim]
                if dim in downscale_factors_per_spatial_dim else 1),
        )
        for dim in dims
    )

    output_chunk = da.coarsen(
        mean_dtype,
        input_array[input_slices],
        axes={
            idim: downscale_factors_per_spatial_dim[dim] if dim in sdims else 1
            for idim, dim in enumerate(dims)
        },
        trim_excess=True,
    )

    output_zarr_arr = zarr.open(
        output_zarr_array_url,
        shape=[int(s) for s in output_shape],
        chunks=[int(cs) for cs in output_chunksizes],
        dtype=input_array.dtype,
        config={'write_empty_chunks': True},
        fill_value=0,
        mode="a",
        **zarr_array_creation_kwargs,
    )

    output_zarr_arr[tuple(
        slice(
            ns_coord[dim],
            ns_coord[dim] + 1,
        )
        if dim in nsdims
        else slice(
            chunk_offset[dim],
            chunk_offset[dim] + chunk_shape[dim],
        )
        for dim in dims
    )] = output_chunk.compute()

    return


def calc_ngff_coordinate_transformations_and_axes(
    stack_properties_res0: dict,
    res_abs_factors: list[dict],
    nsdims: list = None,
):
    
    spacing = stack_properties_res0['spacing']
    origin = stack_properties_res0['origin']
    sdims = list(spacing.keys())
    n_resolutions = len(res_abs_factors)

    coordtfs = [
            [
                {
                    "type": "scale",
                    "scale": [1.0] * len(nsdims)
                    + [
                        float(s * res_abs_factors[res_level][dim])
                        for dim, s in spacing.items()
                    ],
                },
                {
                    "type": "translation",
                    "translation": [0] * len(nsdims)
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
        for dim in nsdims + sdims
    ]

    return coordtfs, axes


def write_sim_to_ome_zarr(
    sim,
    output_zarr_url: str,
    downscale_factors_per_spatial_dim: dict[str, int] = None,
    overwrite: bool = False,
    ngff_version: str = "0.4",
    zarr_array_creation_kwargs: dict = None,
    show_progressbar: bool = True,
    n_batch=1,
    batch_func=None,
    batch_func_kwargs=None,
):
    """
    Write (and compute) a spatial_image (multiview-stitcher flavor)
    to a multiscale NGFF zarr file (v0.4 or v0.5).
    Returns a sim backed by the newly created zarr file.

    If overwrite is False, image data will be read from the zarr file
    and missing pyramid levels will be completed. OME-Zarr metadata
    will be overwritten in any case.

    Note that any transform_key will not be stored in the zarr file.
    However, the returned sim will have the transform_key set as
    in the input sim.

    Parameters
    ----------
    sim : spatial_image
        spatial_image to write
    output_zarr_url : str
        Path to the output zarr file
    downscale_factors_per_spatial_dim : dict, optional
        Downscale factors per spatial dimension to use for
        generating the resolution levels, by default None (no downscaling)
    overwrite : bool, optional
        Whether to overwrite existing data in the output zarr file,
        by default False
    ngff_version : str, optional
        NGFF version to use, by default "0.4"
    zarr_array_creation_kwargs : dict, optional
        Additional keyword arguments to pass to zarr.open
        when creating the zarr arrays, by default None
    show_progressbar : bool, optional
        Whether to show a progress bar (tqdm),
    n_batch : int, optional
        Number of chunks to process in batch when writing
        each resolution level, by default 1
    batch_func : callable, optional
        Function to use for submitting the processing of a batch.
        E.g. misc_utils.process_batch_using_ray, by default None
        (no batch submission, process sequentially)
    batch_func_kwargs : dict, optional
        Additional keyword arguments to pass to batch_func,
        by default None
    
    """

    # if exists and overwrite, remove existing zarr group
    if overwrite and os.path.exists(output_zarr_url):
        print(f"Removing existing {output_zarr_url}...")
        shutil.rmtree(output_zarr_url)

    if zarr_array_creation_kwargs is None:
        zarr_array_creation_kwargs = {}

    # basic handling of OME-Zarr v0.4 and v0.5
    #  - not fully tested for v0.5
    #  - TODO: more relevant differences in v0.5 compared to v0.4?

    zarr_array_creation_kwargs = \
        update_zarr_array_creation_kwargs_for_ngff_version(
            ngff_version, zarr_array_creation_kwargs)

    zarr_group_creation_kwargs = {}
    if ngff_version == "0.4":
        if zarr.__version__ >= "3":
            zarr_group_creation_kwargs = {
                "zarr_format": 2,
            }
    elif ngff_version == "0.5":
        zarr_group_creation_kwargs = {
            "zarr_format": 3,
        }
    else:
        raise ValueError(f"ngff_version {ngff_version} not supported")

    dims = sim.dims
    nsdims = si_utils.get_nonspatial_dims_from_sim(sim)
    sdims = si_utils.get_spatial_dims_from_sim(sim)
    spacing = si_utils.get_spacing_from_sim(sim)
    origin = si_utils.get_origin_from_sim(sim)
    spatial_shape = {
        dim: sim.data.shape[idim]
        for idim, dim in enumerate(dims)
        if dim in sdims
    }

    res_shapes, res_rel_factors, res_abs_factors = \
        msi_utils.calc_resolution_levels(
            spatial_shape,
            downscale_factors_per_spatial_dim=downscale_factors_per_spatial_dim,
        )

    n_resolutions = len(res_shapes)

    coordtfs, axes = calc_ngff_coordinate_transformations_and_axes(
        {
            'spacing': spacing,
            'origin': origin,
            'shape': spatial_shape
        },
        res_abs_factors,
        nsdims=nsdims,
    )

    # parent_res_array = sim.data
    curr_res_array = sim.data  # in case of only one resolution level
    for res_level in range(0, n_resolutions):

        curr_res_array = write_and_return_downsampled_sim(
            curr_res_array,
            dims=dims,
            chunksizes=sim.data.chunksize,
            output_zarr_array_url=f"{output_zarr_url}/{res_level}",
            downscale_factors_per_spatial_dim=res_rel_factors[res_level],
            overwrite=overwrite,
            zarr_array_creation_kwargs=zarr_array_creation_kwargs,
            res_level=res_level,
            show_progressbar=show_progressbar,
            n_batch=n_batch,
            batch_func=batch_func,
            batch_func_kwargs=batch_func_kwargs,
        )

    output_group = zarr.open_group(
        output_zarr_url, mode="a", **zarr_group_creation_kwargs
    )

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
    root = zarr.open_group(zarr_path, mode="r")

    if "omero" in root.attrs:
        omero = root.attrs["omero"]
        ch_coords = [ch["label"] for ch in omero["channels"]]
        sim = sim.assign_coords(c=ch_coords)

    return sim
