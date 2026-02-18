from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr
from dask import delayed

try:
    from readlif.reader import LifFile
except ImportError:
    LifFile = None


def _check_readlif():
    if LifFile is None:
        raise ImportError(
            "readlif is required to read LIF files. "
            "Please install it using `pip install readlif`."
        )


def get_lif_image(filepath, scene_index=0):
    """
    Open a LIF file and return the LifImage for the given scene index.

    Parameters
    ----------
    filepath : str or Path
    scene_index : int

    Returns
    -------
    LifImage
    """
    _check_readlif()
    lif = LifFile(str(filepath))
    return lif.get_image(scene_index)


def get_number_of_scenes(filepath):
    """
    Return the number of images (scenes) in a LIF file.
    """
    _check_readlif()
    lif = LifFile(str(filepath))
    return lif.num_images


def get_spacing_from_lif_image(lif_image):
    """
    Return pixel spacing in micrometers as a dict keyed by spatial dim.

    LifImage.scale is (scale_x, scale_y, scale_z, scale_t) in px/µm.
    Spacing = 1 / scale (µm/px).
    """
    scale_x, scale_y, scale_z, _ = lif_image.scale

    spacing = {}
    if scale_x is not None and scale_x != 0:
        spacing["x"] = 1.0 / scale_x
    else:
        spacing["x"] = 1.0

    if scale_y is not None and scale_y != 0:
        spacing["y"] = 1.0 / scale_y
    else:
        spacing["y"] = 1.0

    if lif_image.nz > 1:
        if scale_z is not None and scale_z != 0:
            spacing["z"] = 1.0 / scale_z
        else:
            spacing["z"] = 1.0

    return spacing


def get_mosaic_tile_origins(lif_image, spacing):
    """
    Return per-tile origins in micrometers.

    mosaic_position entries are (FieldX, FieldY, PosX, PosY),
    where PosX/PosY are stage positions in meters.
    We convert to µm and return as dicts keyed by spatial dim.

    If the image is not a mosaic, returns a single-element list with zeros.
    """
    has_z = "z" in spacing

    if lif_image.n_mosaic <= 1 or not lif_image.mosaic_position:
        origin = {"y": 0.0, "x": 0.0}
        if has_z:
            origin["z"] = 0.0
        return [origin]

    origins = []
    for _fx, _fy, pos_x, pos_y in lif_image.mosaic_position:
        # m -> um
        origin = {
            "y": pos_y * 1e6,
            "x": pos_x * 1e6,
        }
        if has_z:
            origin["z"] = 0.0
        origins.append(origin)

    return origins


def _read_lif_tile(filepath, scene_index, m, z, t, c, shape_yx):
    """
    Read a single (z, t, c, m) plane from a LIF file (for dask delayed).

    Returns a numpy array of shape (y, x).
    """
    import numpy as np

    lif = LifFile(str(filepath))
    lif_image = lif.get_image(scene_index)
    frame = lif_image.get_frame(z=z, t=t, c=c, m=m)
    arr = np.array(frame)
    if arr.shape != tuple(shape_yx):
        arr = arr.reshape(shape_yx)
    return arr


def read_lif_tile_into_xim(filepath, scene_index, m):
    """
    Read a single mosaic tile (index m) from a LIF file into an xarray DataArray.

    The returned DataArray has dimensions ordered as a subset of
    (t, c, z, y, x) depending on what dimensions exist in the image.

    Parameters
    ----------
    filepath : str or Path
    scene_index : int
    m : int
        Mosaic tile index.

    Returns
    -------
    xr.DataArray
    """
    _check_readlif()

    filepath = Path(filepath)
    lif = LifFile(str(filepath))
    lif_image = lif.get_image(scene_index)

    n_t = lif_image.nt
    n_z = lif_image.nz
    n_c = lif_image.channels
    ny = int(lif_image.dims.y)
    nx = int(lif_image.dims.x)
    bit_depth = lif_image.bit_depth[0]
    dtype = np.uint8 if bit_depth == 8 else np.uint16

    has_z = n_z > 1

    # !
    planes = []
    for t in range(n_t):
        c_planes = []
        for c in range(n_c):
            if has_z:
                z_planes = []
                for z in range(n_z):
                    plane = da.from_delayed(
                        delayed(_read_lif_tile)(
                            filepath, scene_index, m, z, t, c, (ny, nx)
                        ),
                        shape=(ny, nx),
                        dtype=dtype,
                    )
                    z_planes.append(plane)
                c_planes.append(da.stack(z_planes, axis=0)) # zyx
            else:
                plane = da.from_delayed(
                    delayed(_read_lif_tile)(
                        filepath, scene_index, m, 0, t, c, (ny, nx)
                    ),
                    shape=(ny, nx),
                    dtype=dtype,
                )
                c_planes.append(plane)  # yx

        planes.append(da.stack(c_planes, axis=0))  # c(z)yx

    data = da.stack(planes, axis=0)  # tc(z)yx

    spacing = get_spacing_from_lif_image(lif_image)

    spatial_dims = list(spacing.keys())
    coords = {}
    for sdim in spatial_dims:
        size = {"x": nx, "y": ny, "z": n_z}[sdim]
        coords[sdim] = np.arange(size) * spacing[sdim]

    coords["t"] = np.arange(n_t, dtype=float)
    coords["c"] = np.arange(n_c)

    dims = ["t", "c", "z", "y", "x"] if has_z else ["t", "c", "y", "x"]

    xim = xr.DataArray(data, dims=dims, coords=coords)

    return xim, spacing
