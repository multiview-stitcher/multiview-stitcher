import itertools
from typing import List, Optional, Tuple, Union

import numpy as np
import spatial_image as si
import xarray as xr
import zarr
from aicsimageio import AICSImage
from tifffile import imread, imwrite
from tqdm import tqdm

from multiview_stitcher import param_utils, spatial_image_utils

METADATA_TRANSFORM_KEY = "affine_metadata"


def read_mosaic_image_into_list_of_spatial_xarrays(path, scene_index=None):
    """
    Read CZI mosaic dataset into xarray containing all information needed for stitching.
    Could eventually be based on https://github.com/spatial-image/spatial-image.
    Use list instead of dict to make sure xarray metadata (coordinates + perhaps attrs)
    are self explanatory for downstream code (and don't depend e.g. on view/tile numbering).


    Comment 202304
    # acisimageio can have problems, namely shape of sim is different from shape of computed sim.data
    # therefore first load sim, then get sim.data, then create spatial image from sim.data and back on disk
    # for multiscale always require zarr format

    """

    aicsim = AICSImage(path, reconstruct_mosaic=False)

    if len(aicsim.scenes) > 1 and scene_index is None:
        from magicgui.widgets import request_values

        scene_index = request_values(
            scene_index={
                "annotation": int,
                "label": "Which scene should be loaded?",
                "options": {"min": 0, "max": len(aicsim.scenes) - 1},
            },
        )["scene_index"]
        aicsim.set_scene(scene_index)
    else:
        scene_index = 0

    sim = aicsim.get_xarray_dask_stack()

    sim = sim.sel(I=scene_index)

    # sim coords to lower case
    sim = sim.rename({dim: dim.lower() for dim in sim.dims})

    # remove singleton Z
    for axis in ["z"]:
        if axis in sim.dims and len(sim.coords[axis]) < 2:
            sim = sim.sel({axis: 0}, drop=True)

    # ensure time dimension is present
    sim = spatial_image_utils.ensure_time_dim(sim)

    spatial_dims = spatial_image_utils.get_spatial_dims_from_sim(sim)

    views = range(len(sim.coords["m"]))

    pixel_sizes = {}
    pixel_sizes["x"] = aicsim.physical_pixel_sizes.X
    pixel_sizes["y"] = aicsim.physical_pixel_sizes.Y
    if "z" in spatial_dims:
        pixel_sizes["z"] = aicsim.physical_pixel_sizes.Z

    view_sims = []
    for iview, view in enumerate(views):
        view_sim = sim.sel(m=view)

        view_sim = spatial_image_utils.get_sim_from_xim(view_sim)

        tile_mosaic_position = aicsim.get_mosaic_tile_position(view)
        origin_values = {
            mosaic_axis: tile_mosaic_position[ima] * pixel_sizes[mosaic_axis]
            for ima, mosaic_axis in enumerate(["y", "x"])
        }

        if "z" in spatial_dims:
            origin_values["z"] = 0

        affine = param_utils.affine_from_translation(
            np.array([origin_values[dim] for dim in spatial_dims])
        )

        affine_xr = xr.DataArray(
            np.stack([affine] * len(view_sim.coords["t"])),
            dims=["t", "x_in", "x_out"],
        )

        spatial_image_utils.set_sim_affine(
            view_sim, affine_xr, METADATA_TRANSFORM_KEY
        )

        view_sim.name = str(iview)

        view_sims.append(view_sim)

    return view_sims


def read_tiff_into_spatial_xarray(
    filename,
    scale=None,
    translation=None,
    affine_transform=None,
    channel_names: Optional[Union[List, Tuple]] = None,
):
    aicsimage = AICSImage(filename)
    sim = aicsimage.get_xarray_dask_stack().squeeze(drop=True)
    sim = sim.rename({dim: dim.lower() for dim in sim.dims})
    sim = spatial_image_utils.ensure_time_dim(sim)

    if "c" not in sim.dims:
        sim = sim.expand_dims(["c"])

    spatial_dims = spatial_image_utils.get_spatial_dims_from_sim(sim)
    sim = sim.transpose(*(("t", "c") + tuple(spatial_dims)))

    if scale is None:
        scale = {ax: 1 for ax in spatial_dims}

    if translation is None:
        translation = {ax: 0 for ax in spatial_dims}

    sim = si.to_spatial_image(
        sim.data,
        dims=sim.dims,
        scale=scale,
        translation=translation,
        c_coords=tuple(channel_names)
        if channel_names is not None
        else list(range(len(sim.coords["c"]))),
    )

    ndim = spatial_image_utils.get_ndim_from_sim(sim)

    if affine_transform is None:
        affine_transform = np.eye(ndim + 1)

    affine_xr = xr.DataArray(
        np.stack([affine_transform] * len(sim.coords["t"])),
        dims=["t", "x_in", "x_out"],
        coords={"t": sim.coords["t"]},
    )

    spatial_image_utils.set_sim_affine(
        sim,
        affine_xr,
        METADATA_TRANSFORM_KEY,
    )

    return sim


def save_sim_as_tif(path, sim):
    """
    Save spatial image as tif file.

    Iterate over non-spatial coordinates and write one "field"
    (i.e. one time point and channel) at a time, because tifffile
    is not thread safe.

    Parameters
    ----------
    path : str
    sim : spatial image
    """

    spatial_dims = spatial_image_utils.get_spatial_dims_from_sim(sim)
    spacing = spatial_image_utils.get_spacing_from_sim(sim, asarray=True)

    sim = sim.transpose(*tuple(["t", "c"] + spatial_dims))

    channels = [
        ch if isinstance(ch, str) else str(ch) for ch in sim.coords["c"].values
    ]

    sim = sim.squeeze(drop=True)

    # imagej needs Z to come before C
    if "z" in sim.dims and "c" in sim.dims:
        axes = list(sim.dims)
        zpos = axes.index("z")
        cpos = axes.index("c")
        axes[zpos] = "c"
        axes[cpos] = "z"
        sim = sim.transpose(*tuple(axes))

    axes = "".join(sim.dims).upper()

    imwrite(
        path,
        shape=sim.shape,
        dtype=sim.dtype,
        imagej=True,
        resolution=tuple([1.0 / s for s in spacing[-2:]]),
        metadata={
            "axes": axes,
            "unit": "um",  # assume um
            "Labels": channels,
        },
    )

    store = imread(path, mode="r+", aszarr=True)
    z = zarr.open(store, mode="r+")

    # iterate over non-spatial dimensions and write one "field" at a time
    nsdims = [dim for dim in sim.dims if dim not in spatial_dims]

    for nsdim_indices in tqdm(
        itertools.product(
            *tuple([range(len(sim.coords[nsdim])) for nsdim in nsdims])
        ),
        total=np.prod([len(sim.coords[nsdim]) for nsdim in nsdims]),
    ):
        sl = tuple([slice(ind, ind + 1) for ind in nsdim_indices]) + tuple(
            [slice(None)] * len(spatial_dims)
        )
        z[sl] = sim.data[sl].compute()

    store.close()

    return
