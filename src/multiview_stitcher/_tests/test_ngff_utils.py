import os
import shutil
import tempfile

import dask.array as da
import ngff_zarr
import numpy as np
import pytest
import zarr
from xarray import DataTree

from multiview_stitcher import (
    io,
    msi_utils,
    ngff_utils,
    sample_data,
)
from multiview_stitcher import spatial_image_utils as si_utils


def _single_scale_msim_from_sim(sim):
    return DataTree.from_dict({"scale0": sim.to_dataset(name="image")})


def _decode_virtual_chunk(virtual_zarr, path, chunk_key):
    zarray = virtual_zarr.array_zarray(path)
    return np.frombuffer(
        virtual_zarr.read_chunk(path, chunk_key),
        dtype=np.dtype(zarray["dtype"]),
    ).reshape(zarray["chunks"])


@pytest.mark.parametrize(
    "ndim, ngff_version, n_batch",
    [(ndim, ngff_version, n_batch)
    for ndim in (2, 3)
    for ngff_version in ("0.4", "0.5")
    for n_batch in (None, 2)
    ],
)
def test_round_trip(ndim, ngff_version, n_batch):
    """Round-trip a sim and msim through OME-Zarr and verify pixel data and
    spatial coordinates are preserved at every resolution level."""
    sim = sample_data.generate_tiled_dataset(
        ndim=ndim,
        overlap=0,
        N_c=1,
        N_t=1,
        tile_size=30,
        tiles_x=1,
        tiles_y=2,
        tiles_z=1,
        spacing_x=1,
        spacing_y=1,
        spacing_z=1,
    )[1]

    if zarr.__version__ < "3" and ngff_version >= "0.5":
        pytest.skip("zarr>=3 required for ngff_version 0.5")

    # sim
    sdims = si_utils.get_spatial_dims_from_sim(sim)

    with tempfile.TemporaryDirectory() as zarr_path:
        ngff_utils.write_sim_to_ome_zarr(
            sim,
            zarr_path,
            overwrite=False,
            ngff_version=ngff_version,
            batch_options={
                "n_batch": n_batch,
            },
        )

        sim_read = ngff_utils.read_sim_from_ome_zarr(zarr_path)

        for dim in sdims:
            assert np.allclose(
                sim.coords[dim].values, sim_read.coords[dim].values
            )

        assert np.allclose(sim.data, sim_read.data)

    # msim
    scale_factors = [2, 2]
    msim = msi_utils.get_msim_from_sim(sim, scale_factors=scale_factors)
    with tempfile.TemporaryDirectory() as zarr_path:
        ngff_multiscales = ngff_utils.msim_to_ngff_multiscales(
            msim, transform_key=io.METADATA_TRANSFORM_KEY
        )
        ngff_zarr.to_ngff_zarr(zarr_path, ngff_multiscales)

        msim_read = ngff_utils.ngff_multiscales_to_msim(
            ngff_zarr.from_ngff_zarr(zarr_path),
            transform_key=io.METADATA_TRANSFORM_KEY,
        )

        assert np.allclose(
            msim[f"scale{len(scale_factors)}/image"].data,
            msim_read[f"scale{len(scale_factors)}/image"].data,
        )

        for ires in range(len(scale_factors) + 1):
            assert np.allclose(
                msi_utils.get_sim_from_msim(msim_read, scale=f"scale{ires}")
                .coords["y"]
                .values,
                msi_utils.get_sim_from_msim(msim, scale=f"scale{ires}")
                .coords["y"]
                .values,
            )

        assert len(msi_utils.get_sorted_scale_keys(msim)) == len(
            msi_utils.get_sorted_scale_keys(msim_read)
        )


@pytest.mark.parametrize("use_dask", [False, True])
@pytest.mark.parametrize(
    "ndim, N_t, N_c",
    [(2, 1, 1), (2, 2, 1), (3, 1, 2), (2, None, None)],
)
def test_ome_zarr_read_write(ndim, N_t, N_c, use_dask):
    """Write a sim to OME-Zarr and read it back, checking that dims, channel
    names and omero window metadata are preserved for various t/c combinations."""
    sim = sample_data.generate_tiled_dataset(
        ndim=ndim,
        overlap=0,
        N_c=N_c if N_c is not None else 1,
        N_t=N_t if N_t is not None else 1,
        tile_size=10,
        tiles_x=2,
        tiles_y=1,
        tiles_z=1,
        spacing_x=0.1,
        spacing_y=0.1,
        spacing_z=2,
    )[1]

    # make sure to also test for the absence of c and t
    if N_c is None:
        sim = sim.drop_vars("c")

    if N_t is None:
        sim = sim.drop_vars("t")

    with tempfile.TemporaryDirectory() as zarr_path:
        sim = ngff_utils.write_sim_to_ome_zarr(sim, zarr_path)

        metadata = zarr.open_group(zarr_path).attrs.asdict()

        if N_c is not None:
            assert "omero" in metadata
            assert "window" in metadata["omero"]["channels"][0]

        sim_read = ngff_utils.read_sim_from_ome_zarr(
            zarr_path,
            use_dask=use_dask,
        )  # , resolution_level=0)

        assert si_utils.is_dask_backed_dataarray(sim_read) == use_dask

        # check dims and channel names are the same
        # assert np.equal(sim.data, sim_read.data).all()
        assert np.array_equal(sim.dims, sim_read.dims)
        # TODO: consider restricting channel coords to string type
        assert np.array_equal(
            [str(v) for v in sim.coords["c"].values],
            [str(v) for v in sim_read.coords["c"].values],
        )


@pytest.mark.parametrize("use_dask", [False, True])
def test_read_msim_from_ome_zarr(use_dask):
    """Verify that read_msim_from_ome_zarr returns a multiscale image with
    correct pixel data, channel names and more than one resolution level."""
    sim = sample_data.generate_tiled_dataset(
        ndim=2,
        overlap=0,
        N_c=2,
        N_t=1,
        tile_size=202,
        tiles_x=2,
        tiles_y=1,
        tiles_z=1,
        spacing_x=0.1,
        spacing_y=0.1,
        spacing_z=2,
        random_data=True,
    )[1]

    with tempfile.TemporaryDirectory() as zarr_path:
        sim = ngff_utils.write_sim_to_ome_zarr(sim, zarr_path)

        msim_read = ngff_utils.read_msim_from_ome_zarr(
            zarr_path, use_dask=use_dask
        )
        sim_read = msi_utils.get_sim_from_msim(msim_read, scale="scale0")

        assert si_utils.is_dask_backed_dataarray(sim_read) == use_dask

        assert np.array_equal(sim.dims, sim_read.dims)
        assert np.allclose(sim.data, sim_read.data)
        assert np.array_equal(
            [str(v) for v in sim.coords["c"].values],
            [str(v) for v in sim_read.coords["c"].values],
        )

        selected_channel = sim.coords["c"].values[1]
        selected_msim = msi_utils.multiscale_sel_coords(
            msim_read,
            {"c": selected_channel},
        )
        selected_sim = msi_utils.get_sim_from_msim(selected_msim)
        assert "c" in selected_sim.coords
        assert str(selected_sim.coords["c"].item()) == str(selected_channel)

        scale_keys = msi_utils.get_sorted_scale_keys(msim_read)
        assert len(scale_keys) > 1
        for scale_key in scale_keys:
            sim_scale = msi_utils.get_sim_from_msim(
                msim_read, scale=scale_key
            )
            assert np.array_equal(
                [str(v) for v in sim.coords["c"].values],
                [str(v) for v in sim_scale.coords["c"].values],
            )


def test_read_sim_from_ome_zarr_backends():
    sim = sample_data.generate_tiled_dataset(
        ndim=2,
        overlap=0,
        N_c=1,
        N_t=1,
        tile_size=32,
        tiles_x=1,
        tiles_y=1,
        tiles_z=1,
        spacing_x=0.1,
        spacing_y=0.1,
        spacing_z=2,
    )[0]

    with tempfile.TemporaryDirectory() as zarr_path:
        ngff_utils.write_sim_to_ome_zarr(sim, zarr_path)

        sim_zarr = ngff_utils.read_sim_from_ome_zarr(zarr_path)
        sim_dask = ngff_utils.read_sim_from_ome_zarr(
            zarr_path, use_dask=True
        )

        assert si_utils.is_xarray_zarr_backed(sim_zarr)
        assert not si_utils.is_dask_backed_dataarray(sim_zarr)
        assert si_utils.is_dask_backed_dataarray(sim_dask)


def test_read_msim_from_ome_zarr_backends():
    sim = sample_data.generate_tiled_dataset(
        ndim=2,
        overlap=0,
        N_c=1,
        N_t=1,
        tile_size=202,
        tiles_x=2,
        tiles_y=1,
        tiles_z=1,
        spacing_x=0.1,
        spacing_y=0.1,
        spacing_z=2,
        random_data=True,
    )[1]

    with tempfile.TemporaryDirectory() as zarr_path:
        ngff_utils.write_sim_to_ome_zarr(sim, zarr_path)

        msim_zarr = ngff_utils.read_msim_from_ome_zarr(zarr_path)
        msim_dask = ngff_utils.read_msim_from_ome_zarr(
            zarr_path, use_dask=True
        )

        sim_zarr = msi_utils.get_sim_from_msim(msim_zarr, scale="scale0")
        sim_dask = msi_utils.get_sim_from_msim(msim_dask, scale="scale0")

        assert si_utils.is_xarray_zarr_backed(sim_zarr)
        assert not si_utils.is_dask_backed_dataarray(sim_zarr)
        assert si_utils.is_dask_backed_dataarray(sim_dask)

        for scale_key in msi_utils.get_sorted_scale_keys(msim_zarr):
            sim_scale = msi_utils.get_sim_from_msim(msim_zarr, scale=scale_key)
            assert sim_scale.attrs.get("_zarr_chunks") is not None


def test_virtual_ome_zarr_metadata_and_numpy_chunk():
    data = np.arange(5 * 6, dtype=np.uint16).reshape(5, 6)
    sim = si_utils.to_spatial_image(
        data,
        dims=["y", "x"],
        scale={"y": 0.5, "x": 0.25},
        translation={"y": 2.0, "x": 4.0},
    )
    virtual_zarr = ngff_utils.VirtualOMEZarr(
        _single_scale_msim_from_sim(sim)
    )

    assert virtual_zarr.root_zgroup() == {"zarr_format": 2}
    root_zattrs = virtual_zarr.root_zattrs()
    assert "multiscales" in root_zattrs
    assert "_ARRAY_DIMENSIONS" not in str(root_zattrs)

    multiscales = root_zattrs["multiscales"][0]
    assert [axis["name"] for axis in multiscales["axes"]] == ["y", "x"]
    assert [axis["unit"] for axis in multiscales["axes"]] == [
        "micrometer",
        "micrometer",
    ]
    assert multiscales["datasets"][0]["path"] == "0"

    zarray = virtual_zarr.array_zarray("0")
    assert zarray["shape"] == [5, 6]
    assert zarray["chunks"] == [5, 6]
    assert zarray["dimension_separator"] == "/"
    assert zarray["compressor"] is None

    decoded = _decode_virtual_chunk(virtual_zarr, "0", "0/0")
    np.testing.assert_array_equal(decoded, data)

    consolidated = virtual_zarr.consolidated_metadata()["metadata"]
    assert ".zattrs" in consolidated
    assert "0/.zarray" in consolidated
    assert consolidated["0/.zattrs"] == {}


def test_virtual_ome_zarr_dask_edge_chunk_padding():
    data = np.arange(5 * 6, dtype=np.uint16).reshape(5, 6)
    dask_data = da.from_array(data, chunks=(3, 4))
    sim = si_utils.to_spatial_image(
        dask_data,
        dims=["y", "x"],
        scale={"y": 1.0, "x": 1.0},
        translation={"y": 0.0, "x": 0.0},
    )
    virtual_zarr = ngff_utils.VirtualOMEZarr(
        _single_scale_msim_from_sim(sim)
    )

    assert virtual_zarr.array_zarray("0")["chunks"] == [3, 4]

    decoded = _decode_virtual_chunk(virtual_zarr, "0", "1/1")
    expected = np.zeros((3, 4), dtype=np.uint16)
    expected[:2, :2] = data[3:5, 4:6]
    np.testing.assert_array_equal(decoded, expected)

    with pytest.raises(KeyError):
        virtual_zarr.read_chunk("0", "2/0")

    with pytest.raises(KeyError):
        virtual_zarr.read_chunk("missing", "0/0")


def test_virtual_ome_zarr_zarr_backed_chunk():
    data = da.from_array(
        np.arange(5 * 6, dtype=np.uint16).reshape(5, 6),
        chunks=(3, 4),
    )
    sim = si_utils.get_sim_from_array(
        data,
        dims=["y", "x"],
        scale={"y": 1.0, "x": 1.0},
        translation={"y": 0.0, "x": 0.0},
    )

    with tempfile.TemporaryDirectory() as zarr_path:
        ngff_utils.write_sim_to_ome_zarr(
            sim,
            zarr_path,
            downscale_factors_per_spatial_dim={"y": 1, "x": 1},
            show_progressbar=False,
        )
        msim = ngff_utils.read_msim_from_ome_zarr(zarr_path)

        virtual_zarr = ngff_utils.VirtualOMEZarr(msim)
        sim_read = msi_utils.get_sim_from_msim(msim, scale="scale0")
        chunk_shape = virtual_zarr.chunk_shapes["0"]
        chunk_key = "/".join("0" for _ in chunk_shape)
        decoded = _decode_virtual_chunk(virtual_zarr, "0", chunk_key)

        indexers = {
            dim: slice(0, chunk_shape[idim])
            for idim, dim in enumerate(sim_read.dims)
        }
        expected = np.asarray(sim_read.isel(indexers).data)
        np.testing.assert_array_equal(decoded, expected)


@pytest.mark.skipif(
    not hasattr(__import__("signal"), "SIGINT"),
    reason="os.kill/SIGINT not available on this platform",
)
def test_virtual_server_stops_when_main_thread_interrupted():
    """
    UNWANTED BEHAVIOUR: When serve_forever() runs in a non-main thread
    (as when a Jupyter / IPyKernel 6+ asyncio kernel executes a notebook cell
    in a thread pool), a SIGINT sent to the process is delivered to the *main*
    thread only.  threading.Event.wait() in the background thread is never
    interrupted, so serve_forever() blocks indefinitely — the server keeps
    running, the port stays bound, and background threads pile up across kernel
    interrupts / restarts.

    The fix must either:
    * register a SIGINT / atexit hook in the main thread that calls server.stop(),
    * or use a timeout-based polling wait so the thread wakes up periodically and
      can check an external stop condition.

    This test FAILS with the current implementation (serve_forever keeps blocking)
    and is expected to PASS after the fix.
    """
    import os
    import signal
    import socket
    import threading
    import time

    STARTUP_WAIT = 0.3    # seconds to let the server start
    STOP_DEADLINE = 2.0   # serve_forever() must stop within this long after the interrupt

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    sim = sample_data.generate_tiled_dataset(
        ndim=2,
        overlap=0,
        N_c=1,
        N_t=1,
        tile_size=10,
        tiles_x=1,
        tiles_y=1,
        tiles_z=1,
    )[0]
    msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
    server = ngff_utils.serve_virtual_ome_zarrs([msim], port=port)

    serve_done = threading.Event()

    def _run_serve():
        """Simulates a Jupyter notebook cell executing in a non-main thread."""
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            # KeyboardInterrupt would only arrive here if the main thread
            # propagated it — which does not happen in Jupyter's thread-pool
            # execution model.
            pass
        finally:
            serve_done.set()

    serve_thread = threading.Thread(target=_run_serve, daemon=True)
    serve_thread.start()

    # Give the server time to start listening.
    time.sleep(STARTUP_WAIT)

    # Simulate Jupyter's "Interrupt Kernel": send SIGINT to the process.
    # SIGINT is delivered to the *main* thread (this thread), NOT to
    # serve_thread.  We catch KeyboardInterrupt here and do nothing further —
    # exactly mirroring IPyKernel cancelling the asyncio task without ever
    # propagating the interrupt to the blocking background thread.
    try:
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(0.05)  # let the signal land
    except KeyboardInterrupt:
        pass  # main thread handles the interrupt; serve_thread keeps blocking

    try:
        # With the current implementation serve_done is never set because
        # serve_forever() is stuck in Event.wait() with no timeout and no
        # external mechanism calls stop().
        stopped_in_time = serve_done.wait(timeout=STOP_DEADLINE)
    finally:
        server.stop()      # always free the port so other tests can run
        serve_thread.join(timeout=1)

    assert stopped_in_time, (
        f"serve_forever() did not stop within {STOP_DEADLINE}s after SIGINT "
        "was delivered to the main thread.  When a Jupyter kernel is "
        "interrupted, the background thread running serve_forever() keeps "
        "blocking indefinitely, leaking the server and its bound port."
    )
    assert not server._thread.is_alive(), (
        "aiohttp server thread is still running after serve_forever() returned."
    )


def test_multiscales_completion():
    """Check that writing without overwrite completes a partially deleted pyramid:
    after removing a resolution level on disk, re-writing fills it in and the
    metadata remains valid."""
    sim = sample_data.generate_tiled_dataset(
        ndim=2,
        overlap=0,
        N_c=1,
        N_t=1,
        tile_size=202,
        tiles_x=2,
        tiles_y=1,
        tiles_z=1,
        spacing_x=0.1,
        spacing_y=0.1,
        spacing_z=2,
    )[1]

    with tempfile.TemporaryDirectory() as zarr_path:
        # write sim to ome zarr
        sim = ngff_utils.write_sim_to_ome_zarr(sim, zarr_path)

        # remove level 1 on disk
        shutil.rmtree(
            os.path.join(zarr_path, "1"),
        )

        # write again
        sim = ngff_utils.write_sim_to_ome_zarr(sim, zarr_path)

        # check that metadata is present
        zarr.open_group(zarr_path, mode="r").attrs.asdict()

        # check that level 1 is now present
        sim_read = ngff_utils.read_sim_from_ome_zarr(
            zarr_path, resolution_level=1
        )

        # check dims and channel names are the same
        # assert np.equal(sim.data, sim_read.data).all()
        assert np.array_equal(sim.dims, sim_read.dims)
        # TODO: consider restricting channel coords to string type
        assert np.array_equal(
            [str(v) for v in sim.coords["c"].values],
            [str(v) for v in sim_read.coords["c"].values],
        )


def test_multiscales_overwrite():
    """Verify that overwrite=True replaces both pixel data and spatial
    coordinates at all resolution levels with those of the new sim."""
    sim1 = si_utils.get_sim_from_array(
        np.zeros((202, 202)),
        translation={"y": 0, "x": 0},
    )
    sim2 = si_utils.get_sim_from_array(
        np.ones((202, 202)),
        translation={"y": 1, "x": 1},
    )

    with tempfile.TemporaryDirectory() as zarr_path:
        # write sim to ome zarr
        ngff_utils.write_sim_to_ome_zarr(sim1, zarr_path)

        # write again
        ngff_utils.write_sim_to_ome_zarr(sim2, zarr_path, overwrite=True)

        # check that read sim is equal to sim2 at
        # all resolution levels
        for res_level in range(2):
            sim_read = ngff_utils.read_sim_from_ome_zarr(
                zarr_path, resolution_level=res_level)
            assert np.min(sim_read.data) == 1
            assert np.max(sim_read.data) == 1

            for dim in sim_read.dims:
                if dim not in si_utils.SPATIAL_DIMS:
                    continue
                assert sim_read.coords[dim].values[0] > 0


@pytest.mark.parametrize("ngff_version", ["0.4", "0.5"])
def test_update_ome_zarr_multiscales_metadata(ngff_version):
    """Write an OME-Zarr with one origin, call update_ome_zarr_multiscales_metadata
    with a new origin, then read back and assert the scale0 translation was
    updated while the multiscales key structure is intact."""
    if zarr.__version__ < "3" and ngff_version >= "0.5":
        pytest.skip("zarr>=3 required for ngff_version 0.5")

    spacing = {"y": 0.5, "x": 0.5}
    translation_orig = {"y": 10.0, "x": 20.0}
    translation_new = {"y": 3.0, "x": 7.0}

    sim = si_utils.get_sim_from_array(
        np.zeros((202, 202)),
        scale=spacing,
        translation=translation_orig,
    )

    with tempfile.TemporaryDirectory() as zarr_path:
        ngff_utils.write_sim_to_ome_zarr(
            sim, zarr_path, ngff_version=ngff_version
        )

        # build an updated msim with different translation
        sim_new = si_utils.get_sim_from_array(
            np.zeros((202, 202)),
            scale=spacing,
            translation=translation_new,
        )
        msim_new = msi_utils.get_msim_from_sim(sim_new, scale_factors=[])
        n_levels = len(
            msi_utils.get_sorted_scale_keys(
                ngff_utils.read_msim_from_ome_zarr(zarr_path)
            )
        )
        # match the number of resolution levels on disk
        msim_new = msi_utils.get_msim_from_sim(
            sim_new,
            scale_factors=[2] * (n_levels - 1),
        )

        ngff_utils.update_ome_zarr_multiscales_metadata(
            zarr_path, msim_new, transform_key=None
        )

        # read back and verify scale0 translation was updated
        sim_read = ngff_utils.read_sim_from_ome_zarr(zarr_path, resolution_level=0)
        sdims = si_utils.get_spatial_dims_from_sim(sim_read)
        for dim in sdims:
            assert np.isclose(
                sim_read.coords[dim].values[0],
                translation_new[dim],
            ), f"Expected {translation_new[dim]} for dim {dim}, got {sim_read.coords[dim].values[0]}"

        # verify that omero metadata (if present) is still intact
        root = zarr.open_group(zarr_path, mode="r")
        all_attrs = dict(root.attrs)
        if ngff_version == "0.5":
            assert "multiscales" in all_attrs.get("ome", {})
        else:
            assert "multiscales" in all_attrs
