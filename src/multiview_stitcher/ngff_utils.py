import atexit
import asyncio
from dataclasses import asdict
from functools import partial
import json
import math
import os, shutil
import signal
import threading

import dask
import ngff_zarr
import numpy as np
import xarray as xr
import zarr
from tqdm import tqdm
from dask import array as da
import dask.diagnostics
from ome_zarr import writer
from xarray import DataTree

from multiview_stitcher import msi_utils, param_utils, misc_utils
from multiview_stitcher import spatial_image_utils as si_utils


def _drop_none_values(value):
    if isinstance(value, dict):
        return {
            key: _drop_none_values(val)
            for key, val in value.items()
            if val is not None
        }
    if isinstance(value, list):
        return [_drop_none_values(val) for val in value]
    return value


def _zarr_dtype(dtype):
    dtype = np.dtype(dtype)
    if dtype.byteorder == "=":
        if dtype.itemsize == 1:
            dtype = dtype.newbyteorder("|")
        else:
            dtype = dtype.newbyteorder("<" if np.little_endian else ">")
    return dtype.str


def _fill_value_for_dtype(dtype):
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.floating):
        return 0.0
    if np.issubdtype(dtype, np.integer):
        return 0
    if np.issubdtype(dtype, np.bool_):
        return False
    return 0


def _regular_chunks_from_dask_chunks(chunks, shape):
    chunk_shape = []
    for axis, (dim_chunks, dim_size) in enumerate(zip(chunks, shape)):
        if not dim_chunks:
            raise ValueError(f"Dimension {axis} has no chunk metadata.")

        regular_chunk = int(dim_chunks[0])
        if regular_chunk <= 0:
            raise ValueError(f"Invalid chunk size {regular_chunk}.")

        interior_chunks = dim_chunks[:-1]
        if any(int(chunk) != regular_chunk for chunk in interior_chunks):
            raise ValueError(
                "Virtual OME-Zarr serving requires regular chunks except "
                f"for final edge chunks; dimension {axis} has chunks "
                f"{dim_chunks}."
            )

        if int(dim_chunks[-1]) > regular_chunk:
            raise ValueError(
                "Virtual OME-Zarr serving requires final edge chunks to be "
                f"no larger than the declared chunk; dimension {axis} has "
                f"chunks {dim_chunks}."
            )

        chunk_shape.append(min(regular_chunk, int(dim_size)))

    return tuple(chunk_shape)


def _chunk_shape_from_sim(sim):
    data = si_utils._get_backend_data(sim)

    if hasattr(data, "chunks"):
        return _regular_chunks_from_dask_chunks(data.chunks, sim.shape)

    preferred_chunks = sim.encoding.get("preferred_chunks")
    if preferred_chunks is not None:
        return tuple(
            min(int(preferred_chunks[dim]), int(sim.sizes[dim]))
            for dim in sim.dims
        )

    zarr_chunks = sim.attrs.get("_zarr_chunks")
    if zarr_chunks is not None and len(zarr_chunks) == sim.ndim:
        return tuple(
            min(int(chunk), int(size))
            for chunk, size in zip(zarr_chunks, sim.shape)
        )

    return tuple(int(size) for size in sim.shape)


def _json_response_dict(obj):
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


class VirtualOMEZarr:
    """
    Read-only virtual OME-Zarr 0.4 / Zarr v2 hierarchy backed by an msim.

    Chunk requests are materialized directly from the source image with
    ``np.asarray(sim.isel(...).data)`` and no temporary Zarr store is written.
    """

    def __init__(self, msim, name="image", compressor=None, omero=None):
        if not msi_utils.is_msim(msim):
            raise TypeError("VirtualOMEZarr expects a MultiscaleSpatialImage.")

        self.msim = msim
        self.name = name
        self.compressor = compressor
        self.omero = omero
        self.scale_keys = msi_utils.get_sorted_scale_keys(msim)
        if not self.scale_keys:
            raise ValueError("msim must contain at least one scale.")

        self.paths = [str(index) for index in range(len(self.scale_keys))]
        self.sims = [
            msi_utils.get_sim_from_msim(msim, scale=scale_key)
            for scale_key in self.scale_keys
        ]
        self.chunk_shapes = {
            path: _chunk_shape_from_sim(sim)
            for path, sim in zip(self.paths, self.sims)
        }

        self._root_zattrs = self._build_root_zattrs()

    def _build_root_zattrs(self):
        # Build OME-Zarr 0.4 .zattrs metadata directly from the sim coordinates
        # without calling ngff_zarr.to_multiscales, which would rechunk the dask
        # arrays and potentially trigger serialisation of large images to disk.
        _DIM_TYPE = {"t": "time", "c": "channel"}
        sdims = si_utils.get_spatial_dims_from_sim(self.sims[0])
        dims = self.sims[0].dims

        axes = [
            _drop_none_values({
                "name": dim,
                "type": _DIM_TYPE.get(dim, "space"),
                "unit": "micrometer" if dim not in _DIM_TYPE else None,
            })
            for dim in dims
        ]

        datasets = []
        for path, sim in zip(self.paths, self.sims):
            spacing = si_utils.get_spacing_from_sim(sim)
            origin = si_utils.get_origin_from_sim(sim)
            scale_values = [
                float(spacing[dim]) if dim in sdims else 1.0
                for dim in dims
            ]
            translation_values = [
                float(origin[dim]) if dim in sdims else 0.0
                for dim in dims
            ]
            datasets.append({
                "path": path,
                "coordinateTransformations": [
                    {"type": "scale", "scale": scale_values},
                    {"type": "translation", "translation": translation_values},
                ],
            })

        metadata = {
            "version": "0.4",
            "name": self.name,
            "axes": axes,
            "datasets": datasets,
        }

        zattrs = {"multiscales": [metadata]}
        if self.omero is not None:
            zattrs["omero"] = (
                asdict(self.omero)
                if hasattr(self.omero, "__dataclass_fields__")
                else self.omero
            )

        return _drop_none_values(zattrs)

    def root_zgroup(self):
        return {"zarr_format": 2}

    def root_zattrs(self):
        return self._root_zattrs

    def array_zattrs(self, path):
        self._get_sim(path)
        return {}

    def array_zarray(self, path):
        sim = self._get_sim(path)
        chunk_shape = self.chunk_shapes[path]
        compressor = (
            self.compressor.get_config()
            if self.compressor is not None
            else None
        )

        return {
            "zarr_format": 2,
            "shape": [int(size) for size in sim.shape],
            "chunks": [int(size) for size in chunk_shape],
            "dtype": _zarr_dtype(sim.dtype),
            "compressor": compressor,
            "fill_value": _fill_value_for_dtype(sim.dtype),
            "order": "C",
            "filters": None,
            "dimension_separator": "/",
        }

    def consolidated_metadata(self):
        metadata = {
            ".zgroup": self.root_zgroup(),
            ".zattrs": self.root_zattrs(),
        }
        for path in self.paths:
            metadata[f"{path}/.zarray"] = self.array_zarray(path)
            metadata[f"{path}/.zattrs"] = self.array_zattrs(path)

        return {
            "zarr_consolidated_format": 1,
            "metadata": metadata,
        }

    def get_json_key(self, key):
        key = key.strip("/")

        if key == ".zgroup":
            return self.root_zgroup()
        if key == ".zattrs":
            return self.root_zattrs()
        if key == ".zmetadata":
            return self.consolidated_metadata()
        if key.endswith("/.zarray"):
            return self.array_zarray(key[: -len("/.zarray")])
        if key.endswith("/.zattrs"):
            return self.array_zattrs(key[: -len("/.zattrs")])

        raise KeyError(key)

    def read_chunk(self, path, chunk_key):
        sim = self._get_sim(path)
        chunk_shape = self.chunk_shapes[path]
        chunk_index = self._parse_chunk_key(path, chunk_key, sim)

        indexers = {}
        for dim, chunk_i, chunk_size, size in zip(
            sim.dims,
            chunk_index,
            chunk_shape,
            sim.shape,
        ):
            start = int(chunk_i * chunk_size)
            stop = min(start + int(chunk_size), int(size))
            indexers[dim] = slice(start, stop)

        chunk = np.asarray(sim.isel(indexers).data)
        chunk = self._pad_edge_chunk(chunk, chunk_shape, sim.dtype)
        chunk = np.ascontiguousarray(chunk)

        if self.compressor is not None:
            return self.compressor.encode(chunk)

        return chunk.tobytes(order="C")

    def _get_sim(self, path):
        if path not in self.paths:
            raise KeyError(path)
        return self.sims[self.paths.index(path)]

    def _parse_chunk_key(self, path, chunk_key, sim):
        parts = chunk_key.strip("/").split("/")
        if len(parts) != sim.ndim:
            raise KeyError(chunk_key)

        try:
            chunk_index = tuple(int(part) for part in parts)
        except ValueError as exc:
            raise KeyError(chunk_key) from exc

        chunk_shape = self.chunk_shapes[path]
        grid_shape = tuple(
            int(math.ceil(size / chunk))
            for size, chunk in zip(sim.shape, chunk_shape)
        )
        if any(
            index < 0 or index >= grid
            for index, grid in zip(chunk_index, grid_shape)
        ):
            raise KeyError(chunk_key)

        return chunk_index

    def _pad_edge_chunk(self, chunk, chunk_shape, dtype):
        if tuple(chunk.shape) == tuple(chunk_shape):
            return chunk.astype(dtype, copy=False)

        padded = np.full(
            chunk_shape,
            _fill_value_for_dtype(dtype),
            dtype=dtype,
        )
        insert = tuple(slice(0, size) for size in chunk.shape)
        padded[insert] = chunk
        return padded


# ---------------------------------------------------------------------------
# Module-level SIGINT routing
# When a VirtualOMEZarrServer is created from the main thread, its _stopped
# event is registered here.  A shared SIGINT handler sets every registered
# event so that serve_forever() unblocks even when it runs in a worker thread
# (e.g. a Jupyter / IPyKernel 6+ thread-pool cell), where signals are
# delivered only to the *main* thread.
# ---------------------------------------------------------------------------
_sigint_lock = threading.Lock()
_sigint_stop_events: list = []
_sigint_prev_handler = None


def _sigint_handler(sig, frame):
    with _sigint_lock:
        for ev in list(_sigint_stop_events):
            ev.set()
    prev = _sigint_prev_handler
    if callable(prev):
        prev(sig, frame)
    else:
        raise KeyboardInterrupt


def _register_sigint_stop_event(event):
    """Register *event* to be set on SIGINT.  No-op when not on main thread."""
    global _sigint_prev_handler
    if threading.current_thread() is not threading.main_thread():
        return
    try:
        with _sigint_lock:
            if event not in _sigint_stop_events:
                _sigint_stop_events.append(event)
            cur = signal.getsignal(signal.SIGINT)
            if cur is not _sigint_handler:
                _sigint_prev_handler = cur
                signal.signal(signal.SIGINT, _sigint_handler)
    except (ValueError, OSError):
        with _sigint_lock:
            if event in _sigint_stop_events:
                _sigint_stop_events.remove(event)


def _unregister_sigint_stop_event(event):
    """Remove *event* from the SIGINT registry; restore the original handler
    when the registry becomes empty."""
    global _sigint_prev_handler
    try:
        with _sigint_lock:
            if event in _sigint_stop_events:
                _sigint_stop_events.remove(event)
            if not _sigint_stop_events:
                cur = signal.getsignal(signal.SIGINT)
                if cur is _sigint_handler and _sigint_prev_handler is not None:
                    try:
                        signal.signal(signal.SIGINT, _sigint_prev_handler)
                    except (ValueError, OSError):
                        pass
                    _sigint_prev_handler = None
    except Exception:
        pass


class VirtualOMEZarrServer:
    def __init__(
        self,
        virtual_zarrs,
        host="127.0.0.1",
        port=8000,
        route_prefix="image",
        max_concurrent_chunks=None,
    ):
        self.virtual_zarrs = {
            f"{route_prefix}_{index}": virtual_zarr
            for index, virtual_zarr in enumerate(virtual_zarrs)
        }
        self.host = host
        self.port = int(port)
        self.max_concurrent_chunks = (
            max_concurrent_chunks
            if max_concurrent_chunks is not None
            else min(4, os.cpu_count() or 1)
        )
        self.urls = [
            f"http://{self.host}:{self.port}/{name}"
            for name in self.virtual_zarrs
        ]
        self._loop = None
        self._runner = None
        self._thread = None
        self._started = threading.Event()
        self._stopped = threading.Event()
        self._start_error = None
        # Register cleanup hooks as early as possible (while still on the
        # calling thread, which may be the main thread).
        atexit.register(self.stop)
        _register_sigint_stop_event(self._stopped)

    def serve_forever(self):
        self.start()
        # Also try here in case serve_forever() is called directly from the
        # main thread (e.g. in a plain script) and __init__ was not.
        _register_sigint_stop_event(self._stopped)
        try:
            self._stopped.wait()
        except KeyboardInterrupt:
            pass
        finally:
            _unregister_sigint_stop_event(self._stopped)
            self.stop()

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return

        self._started.clear()
        self._stopped.clear()
        self._start_error = None
        self._thread = threading.Thread(
            target=self._run_loop_thread,
            daemon=True,
        )
        self._thread.start()
        self._started.wait()
        if self._start_error is not None:
            raise self._start_error

    def stop(self):
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._stopped.set()
        _unregister_sigint_stop_event(self._stopped)
        atexit.unregister(self.stop)

    def _run_loop_thread(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._start_async())
            self._started.set()
            self._loop.run_forever()
        except BaseException as exc:
            self._start_error = exc
            self._started.set()
        finally:
            if self._runner is not None:
                self._loop.run_until_complete(self._runner.cleanup())
            self._loop.close()
            self._stopped.set()

    async def _start_async(self):
        from aiohttp import web

        app = web.Application()
        app["virtual_zarrs"] = self.virtual_zarrs
        app["chunk_semaphore"] = asyncio.Semaphore(
            self.max_concurrent_chunks
        )
        app.router.add_route("*", "/{image_name}", _handle_virtual_zarr_request)
        app.router.add_route(
            "*",
            "/{image_name}/{key:.*}",
            _handle_virtual_zarr_request,
        )

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        print(
            f"Serving virtual OME-Zarrs at http://{self.host}:{self.port} "
            "until interrupted..."
        )


async def _handle_virtual_zarr_request(request):
    from aiohttp import web

    cors_headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
        "Access-Control-Allow-Headers": "*",
    }

    if request.method == "OPTIONS":
        return web.Response(status=204, headers=cors_headers)

    if request.method not in {"GET", "HEAD"}:
        raise web.HTTPMethodNotAllowed(
            request.method,
            ["GET", "HEAD", "OPTIONS"],
            headers=cors_headers,
        )

    image_name = request.match_info["image_name"]
    key = request.match_info.get("key", "").strip("/")
    virtual_zarr = request.app["virtual_zarrs"].get(image_name)
    if virtual_zarr is None:
        raise web.HTTPNotFound(headers=cors_headers)

    try:
        json_obj = virtual_zarr.get_json_key(key)
    except KeyError:
        json_obj = None

    if json_obj is not None:
        payload = _json_response_dict(json_obj)
        return web.Response(
            body=b"" if request.method == "HEAD" else payload,
            content_type="application/json",
            headers={
                **cors_headers,
                "Content-Length": str(len(payload)),
                "Cache-Control": "public, max-age=3600",
            },
        )

    parts = key.split("/", 1)
    if len(parts) != 2:
        raise web.HTTPNotFound(headers=cors_headers)

    path, chunk_key = parts
    try:
        async with request.app["chunk_semaphore"]:
            payload = await asyncio.to_thread(
                virtual_zarr.read_chunk,
                path,
                chunk_key,
            )
    except KeyError:
        raise web.HTTPNotFound(headers=cors_headers)

    return web.Response(
        body=b"" if request.method == "HEAD" else payload,
        content_type="application/octet-stream",
        headers={
            **cors_headers,
            "Content-Length": str(len(payload)),
            "Cache-Control": "public, max-age=3600",
        },
    )


def serve_virtual_ome_zarrs(
    msims,
    host="127.0.0.1",
    port=8000,
    route_prefix="image",
    max_concurrent_chunks=None,
    compressor=None,
):
    virtual_zarrs = [
        VirtualOMEZarr(msim, name=f"{route_prefix}_{index}", compressor=compressor)
        for index, msim in enumerate(msims)
    ]
    return VirtualOMEZarrServer(
        virtual_zarrs,
        host=host,
        port=port,
        route_prefix=route_prefix,
        max_concurrent_chunks=max_concurrent_chunks,
    )


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


def ngff_image_to_sim(ngff_im, transform_key, data=None):
    """
    Convert a ngff_image in-memory representation compatible with NGFF v0.4
    into a spatial_image (multiview-stitcher flavor).
    """

    # Reuse the general sim constructor so zarr-backed reads preserve chunk
    # hints and singleton t/c axes lazily.
    sim = si_utils.get_sim_from_array(
        ngff_im.data if data is None else data,
        dims=ngff_im.dims,
        scale=ngff_im.scale,
        translation=ngff_im.translation,
        transform_key=transform_key,
    )

    sdims = si_utils.get_spatial_dims_from_sim(sim)

    si_utils.set_sim_affine(
        sim,
        param_utils.affine_to_xaffine(
            np.eye(len(sdims) + 1), t_coords=sim.coords["t"].values
        ),
        transform_key=transform_key,
    )

    return sim


def ngff_multiscales_to_msim(ngff_multiscales, transform_key, data_arrays=None):
    """
    Convert a list of ngff_image in-memory representations compatible with NGFF v0.4
    into a multiscale_spatial_image (multiview-stitcher flavor).
    """

    if data_arrays is None:
        data_arrays = [None] * len(ngff_multiscales.images)

    msim_dict = {}
    for iscale, (ngff_im, data_array) in enumerate(
        zip(ngff_multiscales.images, data_arrays)
    ):
        sim = ngff_image_to_sim(
            ngff_im,
            transform_key=transform_key,
            data=data_array,
        )
        curr_scale_msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
        msim_dict[f"scale{iscale}"] = curr_scale_msim["scale0"]

    msim = DataTree.from_dict(msim_dict)

    return msim


def _open_ngff_dataset_arrays(zarr_path, ngff_multiscales):
    # ngff_zarr currently reads image data as dask arrays. For the zarr-backed
    # default path, reuse its parsed metadata but reopen the on-disk arrays.
    return [
        zarr.open_array(os.path.join(zarr_path, dataset.path), mode="r")
        for dataset in ngff_multiscales.metadata.datasets
    ]


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
            
            # make sure output array exists with correct shape and chunks, and with `write_empty_chunks=True`
            zarr.open(
                output_zarr_array_url,
                shape=[int(s) for s in output_shape],
                chunks=[int(cs) for cs in chunksizes],
                dtype=array.dtype,
                config={'write_empty_chunks': True},
                fill_value=0,
                mode="w" if overwrite else "a",
                **zarr_array_creation_kwargs,
            )

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
    batch_options: dict | None = None,
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
    batch_options : dict, optional
        Options for processing chunks in independent batches. Keys:
        - batch_func: Callable, optional
            Function to process each batch of chunks. Inputs:
            1) a list of block_id(s)
            2) function that performs fusion when passed a given block_id
            By default None, in which case the each block is processed sequentially.
        - n_batch: int
            Number of blocks to process in each batch.
            (n_batch>1 only compatible with a provided batch_func). By default 1.
        - batch_func_kwargs: dict, optional
            Additional keyword arguments passed to batch_func.
    
    """

    if batch_options is None:
        batch_options = {}

    n_batch = batch_options.get("n_batch", 1)
    batch_func = batch_options.get("batch_func", None)
    batch_func_kwargs = batch_options.get("batch_func_kwargs", None)

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
    use_dask=False,
):
    """
    Read a multiscale NGFF zarr file (v0.4/v0.5) into a spatial_image
    (multiview-stitcher flavor) at a given resolution level.

    NGFF zarr files v0.4/v0.5 cannot contain affine transformations, so
    an identity transform will be set for the given transform_key.

    Parameters
    ----------
    zarr_path : str or Path
        Path to the zarr file
    resolution_level : int, optional
        Resolution level to read, by default 0 (highest resolution)
    transform_key : str, optional
        By default si_utils.DEFAULT_TRANSFORM_KEY
    use_dask : bool, optional
        If True, keep the image data dask-backed as returned by ngff_zarr.
        By default False, which reopens the on-disk array as a zarr-backed sim.

    Returns
    -------
    spatial_image with transform_key set
    """
    ngff_multiscales = ngff_zarr.from_ngff_zarr(zarr_path)

    if resolution_level >= len(ngff_multiscales.images):
        raise ValueError(
            f"Resolution level {resolution_level} not found in {zarr_path}"
        )

    data = None
    if not use_dask:
        data = _open_ngff_dataset_arrays(zarr_path, ngff_multiscales)[
            resolution_level
        ]

    sim = ngff_image_to_sim(
        ngff_multiscales.images[resolution_level],
        transform_key=transform_key,
        data=data,
    )

    # get channel names from omero metadata if available
    root = zarr.open_group(zarr_path, mode="r")

    if "omero" in root.attrs:
        omero = root.attrs["omero"]
        ch_coords = [ch["label"] for ch in omero["channels"]]
        sim = sim.assign_coords(c=ch_coords)

    return sim


def update_ome_zarr_multiscales_metadata(zarr_path, msim, transform_key):
    """
    Update the multiscales coordinate transformations (scale and translation)
    of an OME-Zarr file on disk using the spacing and origin from the
    resolution levels of an in-memory msim.

    Only the "ome" key (v0.5) or "multiscales" key (v0.4) of the zarr group
    attributes is modified; other metadata (e.g. "omero") is preserved.

    Parameters
    ----------
    zarr_path : str
        Path to the OME-Zarr file on disk.
    msim : multiscale_spatial_image
        In-memory multiscale image whose resolution levels provide the spacing
        and origin (and optionally the translational component of a transform)
        to write back to disk.
    transform_key : str or None
        Transform key from which to extract the translational component of the
        affine. Pass None to use the sim origin only.

    Raises
    ------
    ValueError
        If the on-disk OME-Zarr is not v0.4 or v0.5, or if the number of
        resolution levels in msim does not match the on-disk zarr.
    """
    root = zarr.open_group(zarr_path, mode="a")
    attrs = dict(root.attrs)

    # Detect OME-Zarr version and retrieve the multiscales list
    if "ome" in attrs:
        ngff_version = attrs["ome"].get("version", "0.5")
        if not ngff_version.startswith("0.5"):
            raise ValueError(
                f"On-disk OME-Zarr has unsupported version '{ngff_version}'. "
                "Only v0.4 and v0.5 are supported."
            )
        multiscales = attrs["ome"]["multiscales"]
    elif "multiscales" in attrs:
        multiscales = attrs["multiscales"]
        ngff_version_in_meta = multiscales[0].get("version", "0.4")
        if not ngff_version_in_meta.startswith("0.4"):
            raise ValueError(
                f"On-disk OME-Zarr has unsupported multiscales version "
                f"'{ngff_version_in_meta}'. Only v0.4 and v0.5 are supported."
            )
        ngff_version = "0.4"
    else:
        raise ValueError(
            f"No OME-Zarr multiscales metadata found in {zarr_path}."
        )

    scale_keys = msi_utils.get_sorted_scale_keys(msim)
    n_levels_msim = len(scale_keys)
    n_levels_disk = len(multiscales[0]["datasets"])
    if n_levels_msim != n_levels_disk:
        raise ValueError(
            f"Number of resolution levels in msim ({n_levels_msim}) does not "
            f"match on-disk OME-Zarr ({n_levels_disk})."
        )

    sim0 = msi_utils.get_sim_from_msim(msim, scale=scale_keys[0])
    nsdims = si_utils.get_nonspatial_dims_from_sim(sim0)
    sdims = si_utils.get_spatial_dims_from_sim(sim0)

    for iscale, scale_key in enumerate(scale_keys):
        sim = msi_utils.get_sim_from_msim(msim, scale=scale_key)
        ngff_im = sim_to_ngff_image(sim, transform_key=transform_key)

        new_coordtfs = [
            {
                "type": "scale",
                "scale": [1.0] * len(nsdims)
                + [float(ngff_im.scale[dim]) for dim in sdims],
            },
            {
                "type": "translation",
                "translation": [0.0] * len(nsdims)
                + [float(ngff_im.translation[dim]) for dim in sdims],
            },
        ]
        multiscales[0]["datasets"][iscale]["coordinateTransformations"] = (
            new_coordtfs
        )

    # Write back only the "multiscales" key, leaving all other metadata intact
    if ngff_version.startswith("0.5"):
        # "multiscales" lives inside the "ome" namespace; read-modify-write
        # only that sub-key so that other "ome" entries (e.g. "omero") survive
        ome = dict(root.attrs["ome"])
        ome["multiscales"] = multiscales
        root.attrs["ome"] = ome
    else:
        # "multiscales" is a top-level attr in v0.4
        root.attrs["multiscales"] = multiscales


def read_msim_from_ome_zarr(
    zarr_path,
    transform_key=si_utils.DEFAULT_TRANSFORM_KEY,
    use_dask=False,
):
    """
    Read a multiscale NGFF zarr file (v0.4/v0.5) into a multiscale_spatial_image
    (multiview-stitcher flavor).

    NGFF zarr files v0.4/v0.5 cannot contain affine transformations, so
    an identity transform will be set for the given transform_key.

    Parameters
    ----------
    zarr_path : str or Path
        Path to the zarr file
    transform_key : str, optional
        By default si_utils.DEFAULT_TRANSFORM_KEY
    use_dask : bool, optional
        If True, keep the multiscale image data dask-backed as returned by
        ngff_zarr. By default False, which reopens every scale as zarr-backed.

    Returns
    -------
    multiscale_spatial_image with transform_key set
    """
    ngff_multiscales = ngff_zarr.from_ngff_zarr(zarr_path)

    data_arrays = None
    if not use_dask:
        data_arrays = _open_ngff_dataset_arrays(zarr_path, ngff_multiscales)

    msim = ngff_multiscales_to_msim(
        ngff_multiscales,
        transform_key=transform_key,
        data_arrays=data_arrays,
    )

    # get channel names from omero metadata if available
    root = zarr.open_group(zarr_path, mode="r")
    if "omero" in root.attrs:
        omero = root.attrs["omero"]
        ch_coords = [ch["label"] for ch in omero["channels"]]
        if "c" in msim['scale0']["image"].dims:
            msim = msim.map_over_datasets(
                xr.DataArray.assign_coords,
                kwargs={'c': ch_coords})

    return msim
