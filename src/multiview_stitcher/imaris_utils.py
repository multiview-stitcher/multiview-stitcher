from pathlib import Path

import dask.array as da
import numpy as np

from multiview_stitcher import msi_utils
from multiview_stitcher import spatial_image_utils as si_utils

SPATIAL_DIMS = ("z", "y", "x")
IMARIS_SPATIAL_DIMS = ("x", "y", "z")


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required to read Imaris .ims files. "
            "Please install it, e.g. using 'pip install h5py' "
            "or `pip install multiview-stitcher[imaris]`."
        ) from exc

    return h5py


def _decode_attr(value, dtype=str):
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "SUO":
            items = value.ravel().tolist()
            if all(isinstance(item, bytes) for item in items):
                text = b"".join(items).decode("utf-8")
            else:
                text = "".join(
                    item.decode("utf-8")
                    if isinstance(item, bytes)
                    else str(item)
                    for item in items
                )
        elif value.size == 1:
            return dtype(value.item())
        else:
            text = bytes(value.ravel()).decode("utf-8")
    elif isinstance(value, bytes):
        text = value.decode("utf-8")
    else:
        text = str(value)

    return dtype(text.rstrip("\x00"))


def get_group_path(ires=0, itime=0, ichannel=0):
    return (
        f"DataSet/ResolutionLevel {ires}/"
        f"TimePoint {itime}/Channel {ichannel}"
    )


def get_shape_from_group(filename, group_path):
    h5py = _require_h5py()

    with h5py.File(filename, "r") as f:
        attrs = f[group_path].attrs
        return {
            dim: _decode_attr(attrs[f"ImageSize{dim.upper()}"], dtype=int)
            for dim in SPATIAL_DIMS
        }


def get_spacing_from_ims(filename):
    h5py = _require_h5py()

    with h5py.File(filename, "r") as f:
        attrs = f["DataSetInfo/Image"].attrs
        return {
            dim: (
                _decode_attr(attrs[f"ExtMax{idim}"], dtype=float)
                - _decode_attr(attrs[f"ExtMin{idim}"], dtype=float)
            )
            / _decode_attr(attrs[dim.upper()], dtype=float)
            for idim, dim in enumerate(IMARIS_SPATIAL_DIMS)
        }


def _read_hdf5_block(
    _template_block,
    *,
    filename,
    dataset,
    block_info=None,
):
    h5py = _require_h5py()

    location = block_info[None]["array-location"]
    selection = tuple(slice(start, stop) for start, stop in location)

    with h5py.File(filename, "r", locking=False) as f:
        return np.asarray(f[dataset][selection])


def _dask_array_from_hdf5_dataset(
    filename: str | Path,
    dataset: str,
) -> da.Array:
    h5py = _require_h5py()
    filename = str(filename)

    with h5py.File(filename, "r", locking=False) as f:
        ds = f[dataset]

        if ds.ndim != 3:
            raise ValueError(f"Expected a 3D dataset, got shape {ds.shape}")

        shape = ds.shape
        dtype = ds.dtype
        chunks = ds.chunks if ds.chunks is not None else shape

    template = da.empty(
        shape=shape,
        chunks=chunks,
        dtype=np.dtype([]),
    )

    return template.map_blocks(
        _read_hdf5_block,
        filename=filename,
        dataset=dataset,
        dtype=dtype,
        meta=np.empty((0, 0, 0), dtype=dtype),
    )


def _read_imaris_into_msim_single_field(filename, itime=0, ichannel=0):
    h5py = _require_h5py()

    with h5py.File(filename, "r") as f:
        n_res = len(f["DataSet"].keys())

    spacing0 = get_spacing_from_ims(filename)
    shape0 = get_shape_from_group(
        filename,
        get_group_path(ires=0, itime=itime, ichannel=ichannel),
    )

    sims = []
    for ires in range(n_res):
        group_path = get_group_path(
            ires=ires,
            itime=itime,
            ichannel=ichannel,
        )
        shape = get_shape_from_group(filename, group_path)
        spacing = {
            dim: spacing0[dim] * shape0[dim] / shape[dim]
            for dim in SPATIAL_DIMS
        }
        spatial_selection = tuple(slice(0, shape[dim]) for dim in SPATIAL_DIMS)

        sim = si_utils.get_sim_from_array(
            _dask_array_from_hdf5_dataset(
                filename,
                f"{group_path}/Data",
            )[spatial_selection],
            dims=SPATIAL_DIMS,
            scale=spacing,
            c_coords=[ichannel],
            t_coords=[itime],
        )
        sims.append(sim)

    msim = msi_utils.get_msim_from_sims(sims)
    return msi_utils.correct_multiscale_origins(msim)


def read_imaris_into_msim(filename, itime=0, channels=None):
    """
    Read an Imaris .ims file into a multiscale image (msim) object.

    Parameters
    ----------
    filename : str or Path
        Path to the .ims file.
    itime : int, optional
        Time point to read.
    channels : list of int, optional
        Channels to read. If None, all channels are read.

    Returns
    -------
    msim : multiscale image object
    """
    h5py = _require_h5py()

    with h5py.File(filename, "r") as f:
        timepoint_path = f"DataSet/ResolutionLevel 0/TimePoint {itime}"
        n_channels = len(f[timepoint_path].keys())

    if channels is None:
        channels = range(n_channels)
    channels = list(channels)
    if not channels:
        raise ValueError("channels must contain at least one channel.")

    msims = [
        _read_imaris_into_msim_single_field(
            filename,
            itime=itime,
            ichannel=ichannel,
        )
        for ichannel in channels
    ]

    return msi_utils.concat(msims, dim="c")
