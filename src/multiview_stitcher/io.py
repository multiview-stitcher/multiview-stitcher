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

import contextlib

from multiview_stitcher import spatial_image_utils as si_utils

METADATA_TRANSFORM_KEY = si_utils.DEFAULT_TRANSFORM_KEY


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
    if AICSImage is None:
        raise ImportError(
            "aicsimageio is required to read mosaic CZI files. Please install it using `pip install multiview-stitcher[aicsimageio]` or `pip install aicsimageio`."
        )

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


def get_info_from_multiview_czi(filename):
    """
    Get information from multiview CZI dataset using czifile.

    This code has been taken from MVRegFus and needs to be improved in terms of
    readability and generalizability.

    Small changes wrt to original code:
     - return n_illuminations and n_views.
     - definition of 'origins'
    """

    if czifile is None:
        raise ImportError(
            "czifile is required to read mosaic CZI files. Please install it using `pip install multiview-stitcher[multiview-czi]` or `pip install czifile`."
        )

    from xml.etree import ElementTree as etree

    pathToImage = filename

    infoDict = {}

    imageFile = czifile.CziFile(pathToImage)
    originalShape = imageFile.shape
    metadata = imageFile.metadata()
    imageFile.close()

    metadata = etree.fromstring(metadata)

    e_chs = metadata.findall(
        ".//Dimensions/Channels/Channel/DetectionWavelength"
    )
    channels = list(range(len(e_chs)))
    # wavelengths = [int(float(ch.text)) for ch in e_chs]

    # hopefully more general
    nViews = metadata.findall(".//MultiView")
    multiView = True
    if len(nViews):
        nViews = len(metadata.findall(".//MultiView/View"))
    else:
        nViews = 1
        multiView = False

    nX = int(metadata.findall(".//SizeX")[0].text)
    nY = int(metadata.findall(".//SizeY")[0].text)

    spacing = np.array(
        [
            float(i.text)
            for i in metadata.findall(".//Scaling")[0].findall(".//Value")
        ]
    ) * np.power(10, 6)
    spacing = spacing.astype(np.float64)

    if multiView:

        def count_planes_of_view_in_czifile(self, view):
            """
            get number of zplanes of a given view independently of number of channels and illuminations
            """

            curr_ch = 0
            curr_ill = 0
            i = 0
            for directory_entry in self.filtered_subblock_directory:
                plane_is_wanted = True
                ch_or_ill_changed = False
                for dim in directory_entry.dimension_entries:
                    if (
                        dim.dimension == "V"
                        and view is not None
                        and dim.start != view
                    ):
                        plane_is_wanted = False
                        break

                    if dim.dimension == "C" and curr_ch != dim.start:
                        ch_or_ill_changed = True
                        break

                    if dim.dimension == "I" and curr_ill != dim.start:
                        ch_or_ill_changed = True
                        break

                if plane_is_wanted and not ch_or_ill_changed:
                    i += 1

            return i

        axisOfRotation = np.array(
            [
                float(i)
                for i in metadata.findall(".//AxisOfRotation")[0].text.split(
                    " "
                )
            ]
        )
        axisOfRotation = np.where(axisOfRotation)[0][0]
        centerOfRotation = np.array(
            [
                -float(i)
                for i in metadata.findall(".//CenterPosition")[0].text.split(
                    " "
                )
            ]
        )

        rPositions, xPositions, yPositions, zPositions = [], [], [], []
        nZs = []
        for i in range(nViews):
            baseNode = metadata.findall(".//View[@V='%s']" % i)
            baseNode = baseNode[1] if len(baseNode) == 2 else baseNode[0]
            xPositions.append(float(baseNode.findall(".//PositionX")[0].text))
            yPositions.append(float(baseNode.findall(".//PositionY")[0].text))
            zPositions.append(float(baseNode.findall(".//PositionZ")[0].text))
            rPositions.append(
                float(baseNode.findall(".//Offset")[0].text) / 180.0 * np.pi
            )
            nZs.append(count_planes_of_view_in_czifile(imageFile, i))

        sizes = np.array([[nX, nY, nZs[i]] for i in range(nViews)])
        positions = np.array(
            [xPositions, yPositions, zPositions, rPositions]
        ).swapaxes(0, 1)
        origins = np.array(
            [
                positions[i][:3]
                # - np.array([sizes[i][0] / 2, sizes[i][1] / 2, 0]) * spacing
                - np.array([sizes[i][0] / 2, sizes[i][1] / 2, sizes[i][2] / 2])
                * spacing
                for i in range(len(positions))
            ]
        )

        # infoDict['angles'] = np.array(rPositions)
        infoDict["origins"] = origins
        infoDict["positions"] = positions
        infoDict["centerOfRotation"] = centerOfRotation
        infoDict["axisOfRotation"] = axisOfRotation
        infoDict["sizes"] = sizes
    else:
        nZ = int(metadata.findall(".//SizeZ")[0].text)
        size = np.array([nX, nY, nZ])

        # position = metadata.findall('.//Positions')[3].findall('Position')[0].values()
        position = (
            metadata.findall(".//Positions")[3]
            .findall("Position")[0]
            .attrib.values()
        )
        position = np.array([float(i) for i in position])
        origin = np.array(
            # position[:3] - np.array([size[0] / 2, size[1] / 2, 0]) * spacing
            position[:3]
            - np.array([size[0] / 2, size[1] / 2, size[2] / 2]) * spacing
        )

        infoDict["sizes"] = np.array([size])
        infoDict["positions"] = np.array([position])
        infoDict["origins"] = np.array([origin])

    infoDict["spacing"] = spacing
    infoDict["originalShape"] = np.array(originalShape)
    infoDict["channels"] = channels
    infoDict["n_illuminations"] = infoDict["originalShape"][1]
    infoDict["n_views"] = nViews

    with contextlib.suppress(Exception):
        infoDict["dT"] = float(
            int(metadata.findall("//TimeSpan/Value")[0].text) / 1000
        )

    return infoDict


def read_view_from_multiview_czi(
    input_file, view=None, ch=None, ill=None, z=None, resize=True, order=1
):
    """
    Use czifile to read images (as there's a bug in aicspylibczi20221013, namely that
    neighboring tiles are included (prestitching?) in a given read out tile).
    """

    if czifile is None:
        raise ImportError(
            "czifile is required to read mosaic CZI files. Please install it using `pip install multiview-stitcher[multiview-czi]` or `pip install czifile`."
        )

    if isinstance(input_file, (str, Path)):
        czifile_file = czifile.CziFile(input_file)
    else:
        czifile_file = input_file

    image = []
    for directory_entry in czifile_file.filtered_subblock_directory:
        plane_is_wanted = True
        for dim in directory_entry.dimension_entries:
            if dim.dimension == "V" and view is not None and dim.start != view:
                plane_is_wanted = False
                break

            if dim.dimension == "C" and ch is not None and dim.start != ch:
                plane_is_wanted = False
                break

            if dim.dimension == "I" and ill is not None and dim.start != ill:
                plane_is_wanted = False
                break

            if dim.dimension == "Z" and z is not None and dim.start != z:
                plane_is_wanted = False
                break

        if not plane_is_wanted:
            continue

        subblock = directory_entry.data_segment()
        tile = subblock.data(resize=resize, order=order)

        try:
            image.append(tile)
        except ValueError as e:
            warnings.warn(str(e), UserWarning, stacklevel=1)

    return np.array(image).squeeze()
