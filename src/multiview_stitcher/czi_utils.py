import contextlib
import warnings
from collections import OrderedDict
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr
from dask import delayed

try:
    import czifile
except ImportError:
    czifile = None


def get_czi_shape(filepath):
    """
    Get the shape of a CZI file.
    """

    if czifile is None:
        raise ImportError(
            "czifile is required to read mosaic CZI files. Please install it using `pip install multiview-stitcher[czi]` or `pip install czifile`."
        )

    czifile_file = czifile.CziFile(filepath)

    shape = {
        dim.dimension: ([] if dim.dimension not in ["Y", "X"] else dim.size)
        for dim in czifile_file.filtered_subblock_directory[
            0
        ].dimension_entries
    }
    for directory_entry in czifile_file.filtered_subblock_directory:
        for _idim, dim in enumerate(directory_entry.dimension_entries):
            if dim.dimension in ["Y", "X"]:
                continue
            shape[dim.dimension].append(dim.start)

    shape = {
        k: len(np.unique(v)) if k not in ["Y", "X"] else v
        for k, v in shape.items()
    }

    return OrderedDict(shape)


def get_spacing_from_czi(filepath):
    """
    Get the spacing of a CZI file.
    """

    if czifile is None:
        raise ImportError(
            "czifile is required to read mosaic CZI files. Please install it using `pip install multiview-stitcher[czi]` or `pip install czifile`."
        )

    imageFile = czifile.CziFile(filepath)
    metadata = imageFile.metadata(raw=False)

    entries = metadata["ImageDocument"]["Metadata"]["Scaling"]["Items"][
        "Distance"
    ]
    spacing = {
        e["Id"].lower(): float(e["Value"]) * np.power(10, 6) for e in entries
    }

    return spacing


def get_czi_mosaic_intervals(filepath, scene_index=0):
    """
    Get the mosaic position intervals of a CZI file.
    """

    if czifile is None:
        raise ImportError(
            "czifile is required to read mosaic CZI files. Please install it using `pip install multiview-stitcher[czi]` or `pip install czifile`."
        )

    czifile_file = czifile.CziFile(filepath)

    shape = get_czi_shape(filepath)
    assert "M" in shape

    spacing = get_spacing_from_czi(filepath)
    spacing = {k.upper(): v for k, v in spacing.items()}

    spatial_dims_czi = list(spacing.keys())

    # index of M in shape
    M_index = list(shape.keys()).index("M")

    S_index = list(shape.keys()).index("S") if "S" in shape else None

    intervals = {
        m: {sdim_czi: [np.inf, -np.inf] for sdim_czi in spatial_dims_czi}
        for m in range(shape["M"])
    }

    for directory_entry in czifile_file.filtered_subblock_directory:
        if (
            S_index is not None
            and directory_entry.dimension_entries[S_index].start != scene_index
        ):
            continue
        m = directory_entry.dimension_entries[M_index].start
        for _idim, dim in enumerate(directory_entry.dimension_entries):
            if dim.dimension not in spatial_dims_czi:
                continue
            else:
                intervals[m][dim.dimension][0] = min(
                    intervals[m][dim.dimension][0], dim.start
                )
                intervals[m][dim.dimension][1] = max(
                    intervals[m][dim.dimension][1], dim.start + dim.size - 1
                )

    # multiply by spacing
    for m in intervals:
        for dim in intervals[m]:
            intervals[m][dim] = [v * spacing[dim] for v in intervals[m][dim]]

    return intervals


def get_czi_channel_names(filepath):
    """
    Get the channel names of a CZI file.
    """

    if czifile is None:
        raise ImportError(
            "czifile is required to read mosaic CZI files. Please install it using `pip install multiview-stitcher[czi]` or `pip install czifile`."
        )

    metadata = czifile.CziFile(filepath).metadata(raw=False)

    # N_c = metadata['ImageDocument']['Metadata']['Information']['Image']['SizeC']
    meta_channels = metadata["ImageDocument"]["Metadata"]["Information"][
        "Image"
    ]["Dimensions"]["Channels"]["Channel"]

    if isinstance(meta_channels, list):
        c_coords = [c["Name"] for c in meta_channels]
    else:
        c_coords = [meta_channels["Name"]]

    return c_coords


def read_czi_plane(filename, ide, slices=None):
    """
    Read a single plane from a CZI file.
    """
    czifile_file = czifile.CziFile(filename)

    plane = (
        czifile_file.filtered_subblock_directory[ide]
        .data_segment()
        .data(resize=True, order=1)
    )

    if slices is not None:
        plane = plane[slices]

    czifile_file.close()

    return plane


def read_czi_into_xims(filename, scene_index=0):
    """
    Read the tiles of a CZI file into xarray DataArrays.
    The CZI file is assumed to be a mosaic file. The function reads
    the tiles into xarray DataArrays, with the dimensions and
    coordinates set according to the metadata of the CZI file.
    The function returns a list of xarray DataArrays,
    one for each tile (dimension M).
    """

    if czifile is None:
        raise ImportError(
            "czifile is required to read mosaic CZI files. Please install it using `pip install multiview-stitcher[czi]` or `pip install czifile`."
        )

    shape = get_czi_shape(filename)
    spacing = get_spacing_from_czi(filename)
    c_coords = get_czi_channel_names(filename)

    # upper case
    spacing = {k.upper(): v for k, v in spacing.items()}

    czifile_file = czifile.CziFile(filename)
    dtype = czifile_file.filtered_subblock_directory[0].dtype

    # There's a strange dimension "0" appended to the axes
    # of the data segment, which is not in the metadata.
    # Also, m is missing in the segment axes.
    de_axes = czifile_file.filtered_subblock_directory[0].axes
    if "0" in czifile_file.filtered_subblock_directory[0].axes:
        index = de_axes.index("0")
        slices = [slice(None) for _ in de_axes]
        slices[index] = 0
        slices = tuple(slices)
    else:
        slices = None

    de_axes_valid = [dim for dim in de_axes if dim not in ["0", "M"]]

    m_planes = {}
    for ide, directory_entry in enumerate(
        czifile_file.filtered_subblock_directory
    ):
        pos = {
            dim.dimension: (
                dim.start if dim.dimension not in ["Y", "X"] else 0
            )
            for dim in directory_entry.dimension_entries
        }
        if "S" in pos and pos["S"] != scene_index:
            continue
        data = da.from_delayed(
            delayed(read_czi_plane)(filename, ide, slices),
            shape=[
                directory_entry.stored_shape[i]
                for i, dim in enumerate(de_axes)
                if dim in de_axes_valid
            ],
            dtype=dtype,
        )

        coords = {
            k: [v]
            if k not in ["Y", "X"]
            else np.linspace(
                v,
                v + shape[k] - 1,
                shape[k],
            )
            * spacing[k]
            for k, v in pos.items()
            if k in de_axes_valid
        }

        xim_plane = xr.DataArray(data, dims=de_axes_valid, coords=coords)

        if pos["M"] not in m_planes:
            m_planes[pos["M"]] = []

        m_planes[pos["M"]].append(xim_plane)

    for m in m_planes:
        xim = xr.combine_by_coords([p.rename(None) for p in m_planes[m]])
        xim = xim.assign_coords(C=c_coords)
        m_planes[m] = xim

    xims = [m_planes[m] for m in sorted(m_planes.keys())]

    return xims


def read_view_from_multiview_czi(
    input_file, view=None, ch=None, ill=None, z=None, resize=True, order=1
):
    """
    Use czifile to read images (as there's a bug in aicspylibczi20221013, namely that
    neighboring tiles are included (prestitching?) in a given read out tile).
    """

    if czifile is None:
        raise ImportError(
            "czifile is required to read mosaic CZI files. Please install it using `pip install multiview-stitcher[czi]` or `pip install czifile`."
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
            "czifile is required to read mosaic CZI files. Please install it using `pip install multiview-stitcher[czi]` or `pip install czifile`."
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
