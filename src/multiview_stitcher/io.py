import itertools
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import zarr
from tifffile import TiffFile, imread, imwrite
from tqdm import tqdm

# aicsimageio is an optional dependency
try:
    from aicsimageio import AICSImage
except ImportError:
    AICSImage = None

try:
    import czifile
except ImportError:
    czifile = None


from multiview_stitcher import czi_utils
from multiview_stitcher import spatial_image_utils as si_utils

METADATA_TRANSFORM_KEY = si_utils.DEFAULT_TRANSFORM_KEY


def read_mosaic_into_sims(filepath, scene_index=0):
    """
    Read the tiles of a mosaic dataset into a list of spatial images (sims).
    The function reads the data lazily and sets the tile positions from the metadata.

    If the file is a CZI file, czifile.py is used to read the data.
    Otherwise, aicsimageio is used.

    Parameters
    ----------
    filepath : str or Path
        Path to the mosaic dataset.
    scene_index : int, optional
        Index of the scene to read. Default is 0.

    Returns
    -------
    list
        List of spatial images (sims) for each view / tile.
    """

    filepath = Path(filepath)

    if filepath.suffix == ".czi":
        return read_mosaic_into_sims_czifile(filepath, scene_index=scene_index)

    else:
        return read_mosaic_into_sims_aicsimageio(
            filepath, scene_index=scene_index
        )


def get_number_of_scenes_in_mosaic(filepath):
    """
    Get the number of scenes in a mosaic dataset.

    Parameters
    ----------
    filepath : str or Path
        Path to the mosaic dataset.

    Returns
    -------
    int
        Number of scenes in the dataset.
    """

    filepath = Path(filepath)

    if filepath.suffix == ".czi":
        return czi_utils.get_czi_shape(filepath)["S"]

    else:
        if AICSImage is None:
            raise ImportError(
                "aicsimageio is required to read mosaic files other than CZI. Please install it using `pip install multiview-stitcher[aicsimageio]` or `pip install aicsimageio`."
            )

        aicsim = AICSImage(filepath)
        return len(aicsim.scenes)


def read_mosaic_into_sims_aicsimageio(path, scene_index=0):
    """
    Read the tiles of a mosaic dataset into a list of spatial images (sims).
    The function reads the data lazily and sets the tile positions from the metadata.

    Requires aicsimageio to be installed.

    Parameters
    ----------
    filepath : str or Path
        Path to the mosaic dataset.
    scene_index : int, optional
        Index of the scene to read. Default is 0.

    Comment 202304
    # acisimageio can have problems, namely shape of sim is different from shape of computed sim.data
    # therefore first load sim, then get sim.data, then create spatial image from sim.data and back on disk
    # for multiscale always require zarr format

    """
    if AICSImage is None:
        raise ImportError(
            "aicsimageio is required to read mosaic files other than CZI. Please install it using `pip install multiview-stitcher[aicsimageio]` or `pip install aicsimageio`."
        )

    aicsim = AICSImage(path, reconstruct_mosaic=False)
    aicsim.set_scene(scene_index)

    xim = aicsim.get_xarray_dask_stack()

    xim = xim.sel(I=scene_index)

    # sim coords to lower case
    xim = xim.rename({dim: dim.lower() for dim in xim.dims})

    # remove singleton Z
    for axis in ["z"]:
        if axis in xim.dims and len(xim.coords[axis]) < 2:
            xim = xim.sel({axis: 0}, drop=True)

    spatial_dims = [dim for dim in xim.dims if dim in si_utils.SPATIAL_DIMS]

    views = range(len(xim.coords["m"]))

    pixel_sizes = {}
    pixel_sizes["x"] = aicsim.physical_pixel_sizes.X
    pixel_sizes["y"] = aicsim.physical_pixel_sizes.Y
    if "z" in spatial_dims:
        pixel_sizes["z"] = aicsim.physical_pixel_sizes.Z

    tile_mosaic_positions = aicsim.get_mosaic_tile_positions()

    view_sims = []
    for _iview, (view, tile_mosaic_position) in enumerate(
        zip(views, tile_mosaic_positions)
    ):
        view_xim = xim.sel(m=view)

        origin_values = {
            mosaic_axis: tile_mosaic_position[ima] * pixel_sizes[mosaic_axis]
            for ima, mosaic_axis in enumerate(["y", "x"])
        }

        if "z" in spatial_dims:
            origin_values["z"] = 0

        view_sim = si_utils.get_sim_from_array(
            view_xim.data,
            dims=view_xim.dims,
            scale=pixel_sizes,
            translation=origin_values,
            affine=None,
            transform_key=METADATA_TRANSFORM_KEY,
            c_coords=view_xim.coords["c"].values,
            t_coords=view_xim.coords["t"].values,
        )

        view_sims.append(view_sim)

    return view_sims


def read_mosaic_into_sims_czifile(filename, scene_index=0):
    """
    Read the tiles of a CZI mosaic dataset into a list of sims.
    This function uses czifile.py (instead of aicsimageio) to read the CZI file.
    """

    xims = czi_utils.read_czi_into_xims(filename, scene_index=scene_index)

    dims_to_drop = [
        dim
        for idim, dim in enumerate(xims[0].dims)
        if xims[0].shape[idim] == 1 and dim not in ["C"]
    ]

    xims = [xim.squeeze(dims_to_drop, drop=True) for xim in xims]

    spatial_dims = [
        dim.lower()
        for dim in xims[0].dims
        if dim.lower() in si_utils.SPATIAL_DIMS
    ]

    intervals = czi_utils.get_czi_mosaic_intervals(filename)

    sims = []
    for ixim, xim in enumerate(xims):
        xim = xim.rename({dim: dim.lower() for dim in xim.dims})

        spacing = si_utils.get_spacing_from_sim(xim)

        sim = si_utils.get_sim_from_array(
            xim.data,
            dims=xim.dims,
            scale=spacing,
            translation=si_utils.get_origin_from_sim(xim),
            transform_key=METADATA_TRANSFORM_KEY,
            c_coords=xim.coords["c"].values if "c" in xim.dims else None,
            t_coords=xim.coords["t"].values if "t" in xim.dims else None,
        )

        # set tile positions
        sim = sim.assign_coords(
            {
                sdim: sim.coords[sdim].values
                + intervals[ixim][sdim.upper()][0]
                for sdim in spatial_dims
            }
        )

        sims.append(sim)

    return sims


def read_mosaic_image_into_list_of_spatial_xarrays(filepath, scene_index=0):
    """
    (Deprecated) Read the tiles of a mosaic dataset into a list of spatial images (sims).
    The function reads the data lazily and sets the tile positions from the metadata.

    If the file is a CZI file, czifile.py is used to read the data.
    Otherwise, aicsimageio is used.

    Parameters
    ----------
    filepath : str or Path
        Path to the mosaic dataset.
    scene_index : int, optional
        Index of the scene to read. Default is 0.

    Returns
    -------
    list
        List of spatial images (sims) for each view / tile.
    """
    warnings.warn(
        "read_mosaic_image_into_list_of_spatial_xarrays is deprecated. Use read_mosaic_into_sims instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return read_mosaic_into_sims(filepath, scene_index=scene_index)


def read_tiff_into_spatial_xarray(
    filename: Union[str, Path],
    scale: Optional[dict] = None,
    translation: Optional[dict] = None,
    affine_transform: Optional[Union[np.ndarray, list]] = None,
    dims: Optional[Union[list, tuple]] = None,
    channel_names: Optional[Union[list, tuple]] = None,
):
    """
    Read tiff file into spatial image.

    Parameters
    ----------
    filename : str or Path
    scale : dict, optional
        Pixel spacing
    translation : dict, optional
        Image offset in physical coordinates
    affine_transform : array (ndim+1, ndim+1)
        Affine transform given as homogeneous transformation
        matrix, by default None
    dims : tuple, optional
        Axes dimensions, e.g. ('t', 'c', 'z', 'y', 'x').
        If None, will try to be inferred from metadata.
    channel_names : tuple of str, optional
        Channel names

    Returns
    -------
    SpatialImage (multiview-stitcher flavor)
    """

    with TiffFile(filename) as tif:
        data = tif.asarray()
        axes = tif.series[0].axes

    if dims is None:
        # infer from metadata
        dims = [dim.lower() for dim in axes]

    sim = si_utils.get_sim_from_array(
        data,
        dims=dims,
        scale=scale,
        translation=translation,
        affine=affine_transform,
        transform_key=METADATA_TRANSFORM_KEY,
        c_coords=channel_names,
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

    spatial_dims = si_utils.get_spatial_dims_from_sim(sim)
    spacing = si_utils.get_spacing_from_sim(sim, asarray=True)

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
        sl = [None] * len(sim.dims)
        for nsdim, ind in zip(nsdims, nsdim_indices):
            sl[sim.dims.index(nsdim)] = slice(ind, ind + 1)
        for spatial_dim in spatial_dims:
            sl[sim.dims.index(spatial_dim)] = slice(None)
        sl = tuple(sl)
        z[sl] = sim.data[sl].compute()

    store.close()

    return
