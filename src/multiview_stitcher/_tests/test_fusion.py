import contextlib
import itertools
import os
import tempfile
import warnings

from aiohttp.client_exceptions import ServerDisconnectedError
import cloudpickle
import dask.array as da
import numpy as np
import pytest
import xarray as xr
import zarr
import multiview_stitcher.spatial_image_utils as si_utils
from multiview_stitcher import (
    fusion,
    io,
    misc_utils,
    msi_utils,
    ngff_utils,
    param_utils,
    sample_data,
    transformation,
    weights,
)
from multiview_stitcher.fusion import _core as fusion_core
from multiview_stitcher.io import METADATA_TRANSFORM_KEY


def _sim_to_zarr_backed_sim(sim, path, transform_key):
    sim.data.to_zarr(path, overwrite=True, compute=True)
    zarray = zarr.open_array(path, mode="r")

    return si_utils.get_sim_from_array(
        zarray,
        dims=sim.dims,
        scale=si_utils.get_spacing_from_sim(sim),
        translation=si_utils.get_origin_from_sim(sim),
        affine=si_utils.get_affine_from_sim(sim, transform_key),
        transform_key=transform_key,
        c_coords=sim.coords["c"].values if "c" in sim.coords else None,
        t_coords=sim.coords["t"].values if "t" in sim.coords else None,
    )


def _make_distinct_level_msim():
    # Give scale1 distinct values so tests verify scale selection, not downsampling.
    sim = si_utils.get_sim_from_array(
        np.ones((4, 4)),
        dims=["y", "x"],
        transform_key=METADATA_TRANSFORM_KEY,
    )
    msim = msi_utils.get_msim_from_sim(sim, scale_factors=[{"y": 2, "x": 2}])
    msim["scale1"]["image"] = xr.ones_like(msim["scale1"]["image"]) * 2
    return msim


def test_fuse_sims():
    sims = io.read_mosaic_into_sims(sample_data.get_mosaic_sample_data_path())

    # suppress pandas future warning occuring within xarray.concat
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)

        # test with two channels
        for isim, sim in enumerate(sims):
            sims[isim] = xr.concat([sim] * 2, dim="c").assign_coords(
                c=[sim.coords["c"].data[0], sim.coords["c"].data[0] + "_2"]
            )

    xfused = fusion.fuse(
        sims,
        transform_key=METADATA_TRANSFORM_KEY,
    )

    # check output is dask array and hasn't been converted into numpy array
    assert type(xfused.data) == da.core.Array
    assert xfused.dtype == sims[0].dtype

    # xfused.compute()
    xfused = xfused.compute(scheduler="single-threaded")

    assert xfused.dtype == sims[0].dtype
    assert METADATA_TRANSFORM_KEY in si_utils.get_tranform_keys_from_sim(
        xfused
    )


def test_fuse_zarr_backed_input_stays_zarr_backed_until_chunk_execution(
    monkeypatch,
):
    sim = sample_data.generate_tiled_dataset(
        ndim=2,
        N_c=1,
        N_t=1,
        tile_size=24,
        tiles_x=1,
        tiles_y=1,
        overlap=0,
    )[0]

    observed = []
    serial_read_configs = []
    original_fuse_np = fusion_core.fuse_np
    original_astype = xr.DataArray.astype

    def wrapped_fuse_np(*args, **kwargs):
        sims = kwargs["sims"] if "sims" in kwargs else args[0]
        observed.append(
            all(
                not si_utils.is_xarray_zarr_backed(sim)
                and not si_utils.is_dask_backed_dataarray(sim)
                for sim in sims
            )
        )
        return original_fuse_np(*args, **kwargs)

    def fail_dask_affine_transform(*args, **kwargs):
        raise AssertionError(
            "zarr-backed fusion slices should be materialized to NumPy inside the delayed chunk task"
        )

    @contextlib.contextmanager
    def record_serial_read_config(config):
        serial_read_configs.append(dict(config))
        yield

    monkeypatch.setattr(
        fusion_core,
        "fuse_np",
        wrapped_fuse_np,
    )
    monkeypatch.setattr(
        transformation,
        "dask_image_affine_transform",
        fail_dask_affine_transform,
    )
    monkeypatch.setattr(
        si_utils,
        "_zarr_array_uses_http_store",
        lambda zarray: True,
    )
    monkeypatch.setattr(
        si_utils.zarr.config,
        "set",
        record_serial_read_config,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        zsim = _sim_to_zarr_backed_sim(
            sim,
            os.path.join(tmpdir, "view.zarr"),
            METADATA_TRANSFORM_KEY,
        )

        fused = fusion.fuse(
            [zsim],
            transform_key=METADATA_TRANSFORM_KEY,
            fusion_func=fusion.max_fusion,
            output_chunksize={"y": 8, "x": 8},
        )
        fused.data.compute()

    assert observed and all(observed)


def test_fuse_reuses_spatial_plan_across_channels(monkeypatch):
    sims = sample_data.generate_tiled_dataset(
        ndim=2,
        overlap=0,
        N_c=3,
        N_t=1,
        tile_size=16,
        tiles_x=2,
        tiles_y=1,
        tiles_z=1,
        spacing_x=1.0,
        spacing_y=1.0,
        spacing_z=1.0,
        random_data=True,
        chunksize=8,
    )

    plan_calls = []
    original_build_plan = fusion_core._build_spatial_fusion_plan

    def wrapped_build_plan(*args, **kwargs):
        plan_calls.append(kwargs)
        return original_build_plan(*args, **kwargs)

    monkeypatch.setattr(
        fusion_core,
        "_build_spatial_fusion_plan",
        wrapped_build_plan,
    )

    fusion.fuse(
        sims,
        transform_key=METADATA_TRANSFORM_KEY,
        output_spacing={"y": 2.0, "x": 2.0},
        output_chunksize={"y": 8, "x": 8},
    )

    assert len(plan_calls) == 1


def test_fuse_axis_aligned_translation_plan_skips_generic_overlap(monkeypatch):
    sims = [
        si_utils.get_sim_from_array(
            da.ones((1, 1, 8, 8), chunks=(1, 1, 4, 4)) * value,
            dims=["c", "t", "y", "x"],
            scale={"y": 1.0, "x": 1.0},
            translation={"y": 0.0, "x": x_origin},
            transform_key=METADATA_TRANSFORM_KEY,
        )
        for value, x_origin in [(1, 0.0), (2, 6.0)]
    ]

    def fail_generic_overlap(*args, **kwargs):
        raise AssertionError(
            "axis-aligned translation fusion should not call generic overlap"
        )

    monkeypatch.setattr(
        fusion_core.mv_graph,
        "get_overlap_for_bbs",
        fail_generic_overlap,
    )

    fused = fusion.fuse(
        sims,
        transform_key=METADATA_TRANSFORM_KEY,
        fusion_func=fusion.max_fusion,
        output_chunksize={"y": 4, "x": 4},
    )
    fused_data = np.asarray(fused.data.compute(scheduler="single-threaded"))

    assert fused_data.shape == (1, 1, 8, 14)
    np.testing.assert_array_equal(fused_data[..., :, :6], 1)
    np.testing.assert_array_equal(fused_data[..., :, 6:], 2)


def test_fuse_ome_zarr_dask_backed_matches_zarr_backed_reads():
    sims = sample_data.generate_tiled_dataset(
        ndim=2,
        overlap=0,
        N_c=1,
        N_t=1,
        tile_size=32,
        tiles_x=2,
        tiles_y=1,
        tiles_z=1,
        spacing_x=1.0,
        spacing_y=1.0,
        spacing_z=1.0,
        random_data=True,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        msims_zarr = []
        msims_dask = []

        for isim, sim in enumerate(sims):
            zarr_path = os.path.join(tmpdir, f"sim_{isim}.ome.zarr")
            affine = si_utils.get_affine_from_sim(sim, METADATA_TRANSFORM_KEY)

            ngff_utils.write_sim_to_ome_zarr(sim, zarr_path)

            msim_zarr = ngff_utils.read_msim_from_ome_zarr(
                zarr_path, array_backend="zarr"
            )
            msim_dask = ngff_utils.read_msim_from_ome_zarr(
                zarr_path, array_backend="dask"
            )

            msi_utils.set_affine_transform(
                msim_zarr, affine, transform_key=METADATA_TRANSFORM_KEY
            )
            msi_utils.set_affine_transform(
                msim_dask, affine, transform_key=METADATA_TRANSFORM_KEY
            )

            msims_zarr.append(msim_zarr)
            msims_dask.append(msim_dask)

        fused_zarr = fusion.fuse(
            msims_zarr, transform_key=METADATA_TRANSFORM_KEY
        )
        fused_dask = fusion.fuse(
            msims_dask, transform_key=METADATA_TRANSFORM_KEY
        )

        sim_zarr = msi_utils.get_sim_from_msim(fused_zarr, scale="scale0")
        sim_dask = msi_utils.get_sim_from_msim(fused_dask, scale="scale0")

        fused_zarr_data = np.asarray(sim_zarr.data.compute())
        fused_dask_data = np.asarray(sim_dask.data.compute())

        assert fused_zarr_data.max() > 0
        np.testing.assert_array_equal(fused_dask_data, fused_zarr_data)


@pytest.mark.parametrize("varying_dim", ["c", "t"], ids=["channel", "time"])
def test_fuse_zarr_backed_selected_nonspatial_preserves_identity(varying_dim):
    sim = sample_data.generate_tiled_dataset(
        ndim=2,
        overlap=0,
        N_c=2 if varying_dim == "c" else 1,
        N_t=2 if varying_dim == "t" else 1,
        tile_size=16,
        tiles_x=1,
        tiles_y=1,
        tiles_z=1,
        spacing_x=1.0,
        spacing_y=1.0,
        spacing_z=1.0,
        random_data=True,
    )[0]

    data = np.asarray(sim.data.compute()).copy()
    if varying_dim == "c":
        data[:, 1] = data[:, 0] + 1000
        selected_indices = {"t": 0, "c": 1}
    elif varying_dim == "t":
        data[1] = data[0] + 1000
        selected_indices = {"t": 1, "c": 0}
    else:
        raise ValueError(f"Unsupported non-spatial dim {varying_dim!r}")

    sim = si_utils.get_sim_from_array(
        data,
        dims=sim.dims,
        scale=si_utils.get_spacing_from_sim(sim),
        translation=si_utils.get_origin_from_sim(sim),
        affine=si_utils.get_affine_from_sim(sim, METADATA_TRANSFORM_KEY),
        transform_key=METADATA_TRANSFORM_KEY,
        c_coords=sim.coords["c"].values,
        t_coords=sim.coords["t"].values,
    )

    expected = data[selected_indices["t"], selected_indices["c"]]

    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = os.path.join(tmpdir, "view.ome.zarr")
        ngff_utils.write_sim_to_ome_zarr(sim, zarr_path, overwrite=True)

        msim = ngff_utils.read_msim_from_ome_zarr(zarr_path)
        zsim = msi_utils.get_sim_from_msim(msim, scale="scale0")
        selected = si_utils.sim_sel_coords(
            zsim,
            {
                dim: zsim.coords[dim][[idx]]
                for dim, idx in selected_indices.items()
            },
        )

        selected_data = np.squeeze(np.asarray(selected.data))
        np.testing.assert_array_equal(selected_data, expected)

        fused = fusion.fuse([selected], transform_key=METADATA_TRANSFORM_KEY)
        fused_data = np.squeeze(np.asarray(fused.data.compute()))

        np.testing.assert_array_equal(fused_data, expected)


@pytest.mark.parametrize("varying_dim", ["c", "t"], ids=["channel", "time"])
def test_fuse_zarr_backed_msims_to_ome_zarr_preserves_nonspatial_offsets(
    varying_dim,
):
    sims = sample_data.generate_tiled_dataset(
        ndim=2,
        overlap=8,
        N_c=2 if varying_dim == "c" else 1,
        N_t=2 if varying_dim == "t" else 1,
        tile_size=32,
        tiles_x=2,
        tiles_y=1,
        tiles_z=1,
        spacing_x=1.0,
        spacing_y=1.0,
        spacing_z=1.0,
        random_data=True,
    )

    fixed = []
    for sim in sims:
        data = np.asarray(sim.data.compute()).copy()
        if varying_dim == "c":
            data[:, 1] = data[:, 0] + 1000
        elif varying_dim == "t":
            data[1] = data[0] + 1000
        else:
            raise ValueError(f"Unsupported non-spatial dim {varying_dim!r}")

        fixed.append(
            si_utils.get_sim_from_array(
                data,
                dims=sim.dims,
                scale=si_utils.get_spacing_from_sim(sim),
                translation=si_utils.get_origin_from_sim(sim),
                affine=si_utils.get_affine_from_sim(
                    sim, METADATA_TRANSFORM_KEY
                ),
                transform_key=METADATA_TRANSFORM_KEY,
                c_coords=sim.coords["c"].values,
                t_coords=sim.coords["t"].values,
            )
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        zmsims = []
        for isim, sim in enumerate(fixed):
            view_path = os.path.join(tmpdir, f"view_{isim}.ome.zarr")
            ngff_utils.write_sim_to_ome_zarr(sim, view_path, overwrite=True)
            msim = ngff_utils.read_msim_from_ome_zarr(view_path)
            msi_utils.set_affine_transform(
                msim,
                si_utils.get_affine_from_sim(sim, METADATA_TRANSFORM_KEY),
                transform_key=METADATA_TRANSFORM_KEY,
            )
            zmsims.append(msim)

        output_path = os.path.join(tmpdir, "fused.ome.zarr")
        fused = fusion.fuse(
            zmsims,
            transform_key=METADATA_TRANSFORM_KEY,
            output_zarr_url=output_path,
            zarr_options={"ome_zarr": True, "overwrite": True},
            output_chunksize=16,
        )

        fused_sim = msi_utils.get_sim_from_msim(fused, scale="scale0")
        fused_data = np.squeeze(np.asarray(fused_sim.data.compute()))

        np.testing.assert_allclose(
            fused_data[1] - fused_data[0],
            1000,
            atol=1,
        )


@pytest.mark.parametrize("backend", ["numpy", "dask", "zarr"])
def test_fuse_trim_overlap_keeps_fused_chunk_overlap(backend):
    data = np.ones((10, 10), dtype=np.float32)

    if backend == "numpy":
        array = data
    elif backend == "dask":
        array = da.from_array(data, chunks=(5, 5))
    else:
        array = zarr.array(
            data,
            chunks=(5, 5),
            store=zarr.storage.MemoryStore(),
        )

    sim = si_utils.get_sim_from_array(
        array,
        dims=["y", "x"],
        transform_key=METADATA_TRANSFORM_KEY,
    )

    fuse_kwargs = {
        "images": [sim],
        "transform_key": METADATA_TRANSFORM_KEY,
        "fusion_func": fusion.max_fusion,
        "output_chunksize": {"y": 5, "x": 5},
        "overlap_in_pixels": 1,
    }

    trimmed = fusion.fuse(**fuse_kwargs)
    untrimmed = fusion.fuse(**fuse_kwargs, trim_overlap=False)

    assert trimmed.data.shape[-2:] == (10, 10)
    assert untrimmed.data.shape[-2:] == (14, 14)
    assert untrimmed.data.chunks[-2:] == ((7, 7), (7, 7))
    assert untrimmed.data.compute(scheduler="single-threaded").shape[-2:] == (
        14,
        14,
    )


@pytest.mark.parametrize("backend", ["numpy", "dask", "zarr"])
def test_fuse_singleton_view_slice_preserves_spacing(backend):
    data = np.ones((2, 20), dtype=np.uint16)

    if backend == "numpy":
        array = data
    elif backend == "dask":
        array = da.from_array(data, chunks=data.shape)
    else:
        array = zarr.array(
            data,
            chunks=data.shape,
            store=zarr.storage.MemoryStore(),
        )

    sim = si_utils.get_sim_from_array(
        array,
        dims=["y", "x"],
        scale={"y": 0.3, "x": 0.3},
        translation={"y": 0.0, "x": 0.0},
        transform_key=METADATA_TRANSFORM_KEY,
    )
    output_properties = {
        "origin": {"y": 0.0, "x": -2.7},
        "spacing": {"y": 0.3, "x": 0.3},
        "shape": {"y": 2, "x": 29},
    }

    fused = fusion.fuse(
        [sim],
        transform_key=METADATA_TRANSFORM_KEY,
        fusion_func=fusion.max_fusion,
        interpolation_order=0,
        output_stack_properties=output_properties,
        output_chunksize={"y": 2, "x": 10},
    )
    fused_data = np.squeeze(
        fused.data.compute(scheduler="single-threaded")
    )

    # The first output chunk sees only the first source pixel. Its spacing must
    # still be 0.3 rather than the singleton-coordinate fallback of 1.0.
    np.testing.assert_array_equal(
        fused_data,
        np.tile(
            np.concatenate(
                [np.zeros(9, dtype=np.uint16), np.ones(20, dtype=np.uint16)]
            ),
            (2, 1),
        ),
    )


def test_fuse_grid_aligned_chunk_edge_tolerates_coordinate_roundoff():
    # A large origin makes adjacent coordinate differences slightly noisier
    # than the scale used to construct the full coordinate array.
    origin = 861.5120670572916
    scale = 0.13810709635416665
    data = np.ones((2, 4084), dtype=np.uint16)
    sim = si_utils.get_sim_from_array(
        data,
        dims=["y", "x"],
        scale={"y": scale, "x": scale},
        translation={"y": 0.0, "x": origin},
        transform_key=METADATA_TRANSFORM_KEY,
    )
    inferred_scale = si_utils.get_spacing_from_sim(sim)["x"]
    output_properties = {
        "origin": {"y": 0.0, "x": origin - 9 * inferred_scale},
        "spacing": {"y": inferred_scale, "x": inferred_scale},
        "shape": {"y": 2, "x": 4093},
    }

    fused = fusion.fuse(
        [sim],
        transform_key=METADATA_TRANSFORM_KEY,
        fusion_func=fusion.max_fusion,
        interpolation_order=0,
        output_stack_properties=output_properties,
        output_chunksize={"y": 2, "x": 4084},
    )
    fused_data = np.squeeze(
        fused.data.compute(scheduler="single-threaded")
    )

    np.testing.assert_array_equal(
        fused_data,
        np.tile(
            np.concatenate(
                [np.zeros(9, dtype=np.uint16), np.ones(4084, dtype=np.uint16)]
            ),
            (2, 1),
        ),
    )


def test_materialize_xarray_zarr_backend_retries_server_disconnect(
    monkeypatch,
):
    sim = sample_data.generate_tiled_dataset(
        ndim=2,
        N_c=1,
        N_t=1,
        tile_size=24,
        tiles_x=1,
        tiles_y=1,
        overlap=0,
    )[0]

    recorded_configs = []
    semaphore_entries = []
    attempt_count = 0
    original_asarray = si_utils.np.asarray

    @contextlib.contextmanager
    def record_read_config(config):
        recorded_configs.append(dict(config))
        yield

    class RecordingSemaphore:
        def __enter__(self):
            semaphore_entries.append(True)
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def flaky_asarray(arg, *args, **kwargs):
        nonlocal attempt_count
        attempt_count += 1

        if attempt_count == 1:
            raise ServerDisconnectedError()

        return original_asarray(arg, *args, **kwargs)

    with tempfile.TemporaryDirectory() as tmpdir:
        zsim = _sim_to_zarr_backed_sim(
            sim,
            os.path.join(tmpdir, "view.zarr"),
            METADATA_TRANSFORM_KEY,
        )

        monkeypatch.setattr(
            si_utils,
            "_zarr_array_uses_http_store",
            lambda zarray: True,
        )
        monkeypatch.setattr(
            si_utils.zarr.config,
            "set",
            record_read_config,
        )
        monkeypatch.setattr(
            si_utils,
            "_HTTP_ZARR_MATERIALIZATION_SEMAPHORE",
            RecordingSemaphore(),
        )
        monkeypatch.setattr(
            si_utils.np,
            "asarray",
            flaky_asarray,
        )

        result = si_utils._materialize_xarray_zarr_backend(
            zsim,
            max_retries=2,
        )

    assert isinstance(result, np.ndarray)
    assert len(recorded_configs) == 2
    assert attempt_count >= 2
    assert len(semaphore_entries) == 2
    assert all(
        config.get("async.concurrency") == si_utils.HTTP_ZARR_ASYNC_CONCURRENCY
        for config in recorded_configs
    )


def test_fuse_rejects_mixed_sims_and_msims():
    msim = _make_distinct_level_msim()
    sim = msi_utils.get_sim_from_msim(msim)

    with pytest.raises(ValueError, match="same kind"):
        fusion.fuse(
            [msim, sim],
            transform_key=METADATA_TRANSFORM_KEY,
        )


def test_fuse_msims_accepts_images_keyword_and_deprecates_sims_keyword():
    msim = _make_distinct_level_msim()
    fuse_kwargs = {
        "transform_key": METADATA_TRANSFORM_KEY,
        "fusion_func": fusion.max_fusion,
        "output_chunksize": 64,
        "output_shape": {"y": 202, "x": 202},
    }

    fused = fusion.fuse(images=[msim], **fuse_kwargs)

    with pytest.warns(DeprecationWarning, match="Use images=... instead"):
        fused_alias = fusion.fuse(sims=[msim], **fuse_kwargs)

    assert msi_utils.get_sorted_scale_keys(fused) == ["scale0", "scale1"]
    assert msi_utils.get_sorted_scale_keys(fused_alias) == ["scale0", "scale1"]
    assert np.max(fused["scale0/image"].data.compute()) == 1
    assert np.max(fused_alias["scale1/image"].data.compute()) == 2


def test_fuse_msims_returns_msim_from_suitable_input_levels():
    msim = _make_distinct_level_msim()

    # Lazy msim fusion should assemble each output level from the matching input level.
    fused = fusion.fuse(
        [msim],
        transform_key=METADATA_TRANSFORM_KEY,
        fusion_func=fusion.max_fusion,
        output_chunksize=64,
        output_shape={"y": 202, "x": 202},
    )

    assert msi_utils.get_sorted_scale_keys(fused) == ["scale0", "scale1"]
    assert METADATA_TRANSFORM_KEY in fused["scale0"].data_vars
    assert METADATA_TRANSFORM_KEY in fused["scale1"].data_vars
    assert np.max(fused["scale0/image"].data.compute()) == 1
    assert np.max(fused["scale1/image"].data.compute()) == 2


def test_fuse_msims_to_zarr_uses_suitable_input_level():
    msim = _make_distinct_level_msim()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "fused.zarr")
        # A requested spacing of 2 should select scale1, whose values are all 2.
        fused = fusion.fuse(
            [msim],
            transform_key=METADATA_TRANSFORM_KEY,
            fusion_func=fusion.max_fusion,
            output_spacing={"y": 2, "x": 2},
            output_origin={"y": 0.5, "x": 0.5},
            output_shape={"y": 2, "x": 2},
            output_chunksize=2,
            output_zarr_url=output_path,
            zarr_options={"ome_zarr": False},
        )

        assert msi_utils.is_msim(fused)
        assert msi_utils.get_sorted_scale_keys(fused) == ["scale0"]
        assert METADATA_TRANSFORM_KEY in fused["scale0"].data_vars
        assert np.max(fused["scale0/image"].data.compute()) == 2


def test_fuse_msims_to_ome_zarr_returns_msim():
    msim = _make_distinct_level_msim()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "fused.ome.zarr")
        fused = fusion.fuse(
            [msim],
            transform_key=METADATA_TRANSFORM_KEY,
            fusion_func=fusion.max_fusion,
            output_spacing={"y": 2, "x": 2},
            output_origin={"y": 0.5, "x": 0.5},
            output_shape={"y": 2, "x": 2},
            output_chunksize=2,
            output_zarr_url=output_path,
            zarr_options={"ome_zarr": True},
        )

        assert msi_utils.is_msim(fused)
        assert METADATA_TRANSFORM_KEY in fused["scale0"].data_vars
        assert si_utils.is_dask_backed_dataarray(fused["scale0/image"])
        assert np.max(fused["scale0/image"].data.compute()) == 2


def test_fuse_msims_with_fractional_intrinsic_translation(monkeypatch):
    # Fractional translations are still axis-aligned translations, so they
    # should use the translation overlap planner rather than the generic affine
    # overlap code.
    def fail_generic_overlap(*args, **kwargs):
        raise AssertionError(
            "fractional translations should use the translation overlap path"
        )

    monkeypatch.setattr(
        fusion_core.mv_graph,
        "get_overlap_for_bbs",
        fail_generic_overlap,
    )

    a = 8.5
    tile_translations = [
        {"y": 0, "x": 0},
        {"y": a, "x": 0},
        {"y": 0, "x": a},
        {"y": a, "x": a},
    ]
    tile_arrays = [
        np.full((2, 10, 10), iview + 1, dtype=np.uint16)
        for iview in range(len(tile_translations))
    ]

    msims = []
    for tile_array, tile_translation in zip(tile_arrays, tile_translations):
        sim = si_utils.get_sim_from_array(
            tile_array,
            dims=["c", "y", "x"],
            scale={"y": 1, "x": 1},
            translation=tile_translation,
            transform_key=METADATA_TRANSFORM_KEY,
            c_coords=["DAPI", "GFP"],
        )
        msims.append(msi_utils.get_msim_from_sim(sim, scale_factors=[]))

    fused_msim = fusion.fuse(
        images=msims,
        transform_key=METADATA_TRANSFORM_KEY,
        output_chunksize={"y": 5, "x": 5},
    )
    fused = msi_utils.get_sim_from_msim(fused_msim, scale="scale0")

    fused_data = fused.data.compute()

    # The last valid tile center is at 17.5, so an output grid with origin 0
    # and spacing 1 should stop at center 17 and not create an empty border at
    # center 18.
    assert fused.sizes["y"] == 18
    assert fused.sizes["x"] == 18
    assert np.max(fused_data) == 4
    assert np.min(fused_data) > 0


@pytest.mark.parametrize("ome_zarr", [False, True])
def test_fuse_sims_to_zarr_returns_sim(ome_zarr):
    sim = msi_utils.get_sim_from_msim(_make_distinct_level_msim())

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(
            tmpdir,
            "fused.ome.zarr" if ome_zarr else "fused.zarr",
        )
        fused = fusion.fuse(
            [sim],
            transform_key=METADATA_TRANSFORM_KEY,
            fusion_func=fusion.max_fusion,
            output_shape={"y": 4, "x": 4},
            output_chunksize=2,
            output_zarr_url=output_path,
            zarr_options={"ome_zarr": ome_zarr},
        )

        assert not msi_utils.is_msim(fused)
        assert np.max(fused.data.compute()) == 1


@pytest.mark.parametrize(
    "ndim, weights_func",
    [
        (2, None),
        (2, weights.content_based),
        (3, None),
        (3, weights.content_based),
    ],
)
def test_multi_view_fusion(ndim, weights_func):
    nviews = 3

    sims = [
        sample_data.generate_tiled_dataset(
            ndim=ndim,
            overlap=0,
            N_c=1,
            N_t=1,
            tile_size=20,
            tiles_x=1,
            tiles_y=1,
            tiles_z=1,
            spacing_x=1,
            spacing_y=1,
            spacing_z=1,
        )[0]
        for _ in range(nviews)
    ]

    # prepare assertion
    for _, sim in enumerate(sims):
        sim.data += 1

    fused = fusion.fuse(
        sims[:],
        transform_key=METADATA_TRANSFORM_KEY,
        weights_func=weights_func,
        weights_func_kwargs=(
            {"sigma_1": 1, "sigma_2": 2}
            if weights_func == weights.content_based
            else None
        ),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        fused.data = da.to_zarr(
            fused.data,
            os.path.join(tmpdir, "fused_sim.zarr"),
            overwrite=True,
            return_stored=True,
            compute=True,
        )

        mfused = msi_utils.get_msim_from_sim(fused, scale_factors=[])

        fused_path = os.path.join(tmpdir, "fused.zarr")
        mfused.to_zarr(fused_path, mode="w")

        mfused = msi_utils.multiscale_spatial_image_from_zarr(fused_path)

        assert fused.data.min() > 0


def test_fused_field_coverage():
    scale = {"y": 2, "x": 0.5}
    affine = param_utils.affine_from_translation([1000, -1000])

    sims = []
    N_x, N_y = 3, 3
    for ix in range(N_x):
        for iy in range(N_y):
            sims.append(
                si_utils.get_sim_from_array(
                    da.ones((20, 20), chunks=(10, 10)) + ix + iy,
                    dims=["y", "x"],
                    scale=scale,
                    translation={
                        "y": iy * 20 * scale["y"] - 100,
                        "x": ix * 20 * scale["x"] + 100,
                    },
                    affine=affine,
                    transform_key=METADATA_TRANSFORM_KEY,
                )
            )

    fused = fusion.fuse(
        sims,
        transform_key=METADATA_TRANSFORM_KEY,
        output_chunksize=13,
        output_spacing=scale,
    )
    fusedc = fused.data.compute(scheduler="single-threaded")

    assert np.min(fusedc) > 0


def test_fused_field_slice():
    """
    Make sure that slice in fused output
      - is properly aligned with the input
      - only requires input data from the equivalent input slice
    """

    # construct array that will complain if requested for
    # chunks that don't have z index 1

    def provide_only_slice(x, imval, block_info=None):
        block_slices = block_info[None]["array-location"]
        if block_slices[0][0] != 1:
            raise ValueError(
                "This part of the input array shouldn't be required"
            )
        else:
            return np.ones(x.shape) * imval

    imval = 1.0
    dim = da.map_blocks(
        provide_only_slice,
        da.empty((5, 50, 100), chunks=(1, 50, 50)),
        dtype=np.float32,
        imval=imval,
    )

    sdims = ["z", "y", "x"]
    spacing = {"z": 3.5, "y": 2.5, "x": 4.5}
    affine_translation = {"z": 1.3, "y": 1, "x": 2}
    sim = si_utils.get_sim_from_array(
        dim,
        dims=sdims,
        scale=spacing,
        transform_key=METADATA_TRANSFORM_KEY,
        affine=param_utils.affine_from_translation(
            [affine_translation[dim] for dim in sdims]
        ),
    )

    output_stack_properties = {
        "spacing": spacing,
        "origin": {
            dim: t + 1 * spacing[dim] for dim, t in affine_translation.items()
        },
        "shape": {"z": 1, "y": 40, "x": 70},
    }

    fused = fusion.fuse(
        [sim],
        transform_key=METADATA_TRANSFORM_KEY,
        interpolation_order=1,
        output_stack_properties=output_stack_properties,
    ).compute()

    assert not any(fused.data.flatten() - imval)


def test_3D_single_plane_fusion():
    """
    Make sure that 3D single plane fusion works
    (i.e. the z axis of the input has length 1)
    """
    sim = si_utils.get_sim_from_array(
        np.ones((1, 10, 10)),
        dims=["z", "y", "x"],
        transform_key=METADATA_TRANSFORM_KEY,
    )

    # fails if output_chunksize[z] != 1 because the
    # weight calculation assumes shape > 1
    fusion.fuse(
        [sim],
        output_shape={"z": 2, "y": 10, "x": 10},
        output_chunksize={"z": 1, "y": 10, "x": 10},
        transform_key=METADATA_TRANSFORM_KEY,
    ).compute(scheduler="single-threaded")


def test_blending_widths():
    """
    Simple test to check that the blending widths are taken into account
    """
    sims = io.read_mosaic_into_sims(sample_data.get_mosaic_sample_data_path())

    fused_small_bw = (
        fusion.fuse(
            sims,
            transform_key=METADATA_TRANSFORM_KEY,
            blending_widths={dim: 0.001 for dim in ["y", "x"]},
        )
        .compute()
        .data
    )

    fused_large_bw = (
        fusion.fuse(
            sims,
            transform_key=METADATA_TRANSFORM_KEY,
            blending_widths={dim: 10 for dim in ["y", "x"]},
        )
        .compute()
        .data
    )

    # make sure the fusion results are different
    assert not np.allclose(fused_small_bw, fused_large_bw)


def test_large_shape_fusion():
    """
    Make sure that arrays with shape > uin16 limit can be fused
    """
    sims = [
        si_utils.get_sim_from_array(
            np.ones((2, 50000)),
            dims=["y", "x"],
            transform_key=METADATA_TRANSFORM_KEY,
        )
        for _ in range(2)
    ]

    sims[1] = sims[1].assign_coords(x=sims[1].coords["x"] + 50000)

    # fails if output_chunksize[z] != 1 because the
    # weight calculation assumes shape > 1
    fused = fusion.fuse(
        sims,
        transform_key=METADATA_TRANSFORM_KEY,
    )

    assert fused.data.shape[-1] > 60000


@pytest.mark.parametrize("backend", ["dask", "zarr", "numpy"])
@pytest.mark.parametrize(
    "input_chunksize",
    [
        {"y": 5, "x": 5},
        {"z": 4, "y": 5, "x": 5},
        {"z": 1, "y": 5, "x": 5},
        {"y": None, "x": None},
    ],
)
def test_fusion_chunksizes(input_chunksize, backend):
    ndim = len(input_chunksize)
    output_chunksize = {
        dim: cs * 2 if cs is not None else 5
        for dim, cs in input_chunksize.items()
    }

    if backend == "numpy" and input_chunksize["x"] is not None:
        pytest.skip("numpy backend only applies to non-chunked input")

    with tempfile.TemporaryDirectory() as tmpdir:
        sims = []
        for isim in range(2):
            shape = [2] + [10] * len(input_chunksize)
            dims = ["c"] + list(input_chunksize.keys())

            if backend == "dask":
                array = da.zeros(
                    shape,
                    chunks=[1] + list(input_chunksize.values()),
                )
                sim = si_utils.get_sim_from_array(array, dims=dims)
            elif backend == "zarr":
                if input_chunksize["x"] is None:
                    pytest.skip("zarr backend requires explicit source chunks")

                zarray = zarr.open_array(
                    os.path.join(tmpdir, f"input_{isim}.zarr"),
                    mode="w",
                    shape=tuple(shape),
                    chunks=tuple([1] + list(input_chunksize.values())),
                    dtype=np.float32,
                )
                sim = si_utils.get_sim_from_array(zarray, dims=dims)
            else:
                array = np.zeros(shape)
                sim = si_utils.get_sim_from_array(array, dims=dims)

            sims.append(sim)

        for set_output_chunksize in [True, False]:
            fused = fusion.fuse(
                sims,
                transform_key=METADATA_TRANSFORM_KEY,
                output_chunksize=(
                    output_chunksize if set_output_chunksize else None
                ),
            )

            if set_output_chunksize:
                expected_chunksize = output_chunksize
            else:
                if (
                    backend in ["dask", "zarr"]
                    and input_chunksize["x"] is not None
                ):
                    expected_chunksize = input_chunksize
                else:
                    expected_chunksize = {
                        dim: min(
                            fused.shape[-ndim + idim],
                            si_utils.get_default_spatial_chunksizes(ndim)[dim],
                        )
                        for idim, dim in enumerate(
                            si_utils.SPATIAL_DIMS[-ndim:]
                        )
                    }
            assert all(
                fused.data.chunksize[-ndim + idim] == expected_chunksize[dim]
                for idim, dim in enumerate(si_utils.SPATIAL_DIMS[-ndim:])
            )


def test_fuse_to_zarr():

    sims = io.read_mosaic_into_sims(sample_data.get_mosaic_sample_data_path())

    fuse_kwargs = {
        "images": sims,
        "transform_key": METADATA_TRANSFORM_KEY,
    }

    with tempfile.TemporaryDirectory() as tmpdir:

        output_path = os.path.join(tmpdir, "fused.zarr")

        for batch_func in [
            None,
            # fusion.process_batch_using_ray, # leads to obscure error in CI, probably https://github.com/ray-project/ray/issues/55255
        ]:

            fused = fusion.fuse(
                output_zarr_url=output_path,
                zarr_options={
                    "ome_zarr": False,
                },
                batch_options={
                    "n_batch": 2,
                    "batch_func": batch_func,
                },
                **fuse_kwargs,
            )

            assert fused.max().compute() > 0

        fused = fusion.fuse(
            output_zarr_url=output_path,
            zarr_options={
                "ome_zarr": False,
            },
            batch_options={
                "n_batch": 2,
            },
            **fuse_kwargs,
        )

        assert fused.max().compute() > 0

        output_path = os.path.join(tmpdir, "fused.ome.zarr")

        fused = fusion.fuse(
            output_zarr_url=output_path,
            zarr_options={
                "ome_zarr": True,
                # "ngff_version": "0.4",  # optional, defaults to 0.4
            },
            batch_options={
                "n_batch": 2,
            },
            **fuse_kwargs,
        )

        fused = ngff_utils.read_sim_from_ome_zarr(
            output_path,
        )

        assert fused.max().compute() > 0


@pytest.mark.parametrize("array_backend", ["dask", "zarr"])
def test_fuse_with_cupy_backend(array_backend):
    try:
        import cupy as cp
    except ImportError:
        pytest.skip("CuPy not available")

    sims = sample_data.generate_tiled_dataset(
        ndim=2,
        N_c=1,
        N_t=1,
        tile_size=20,
        tiles_x=2,
        tiles_y=1,
        overlap=5,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        if array_backend == "zarr":
            sims = [
                _sim_to_zarr_backed_sim(
                    sim,
                    os.path.join(tmpdir, f"view_{i}.zarr"),
                    METADATA_TRANSFORM_KEY,
                )
                for i, sim in enumerate(sims)
            ]

        fused = fusion.fuse(
            sims,
            transform_key=METADATA_TRANSFORM_KEY,
            backend="cupy",
        ).compute(scheduler="single-threaded")

    assert fused.dtype == sims[0].dtype
    assert isinstance(fused.data, np.ndarray)


def _small_zarr_backed_sim(tmpdir, name, value, tx=0.0, c_coord=0, t_coord=0.0):
    """Build a tiny zarr-backed sim (t=1, c=1) at x-offset ``tx``."""
    sim = si_utils.get_sim_from_array(
        np.full((1, 1, 16, 16), value, dtype=np.uint16),
        dims=["t", "c", "y", "x"],
        scale={"y": 1.0, "x": 1.0},
        translation={"y": 0.0, "x": float(tx)},
        transform_key=METADATA_TRANSFORM_KEY,
        c_coords=[c_coord],
        t_coords=[t_coord],
    )
    return _sim_to_zarr_backed_sim(
        sim, os.path.join(tmpdir, name), METADATA_TRANSFORM_KEY
    )


def test_fuse_concat_c_zarr_backed_sims(monkeypatch):
    """Fusing a zarr-backed sim assembled by concat along c stays on the lazy
    (input_is_zarr) path and produces the correct per-channel result."""
    serialize_calls = []
    original_serialize = si_utils.serialize_zarr_backed_sim

    def counting_serialize(sim):
        serialize_calls.append(sim)
        return original_serialize(sim)

    monkeypatch.setattr(
        si_utils, "serialize_zarr_backed_sim", counting_serialize
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        s0 = _small_zarr_backed_sim(tmpdir, "c0.zarr", 10, c_coord=0)
        s1 = _small_zarr_backed_sim(tmpdir, "c1.zarr", 30, c_coord=1)

        combined = si_utils.concat([s0, s1], dim="c")
        assert si_utils.is_xarray_zarr_backed(combined)

        fused = fusion.fuse([combined], transform_key=METADATA_TRANSFORM_KEY)
        result = np.asarray(fused.data)

    # serialize_zarr_backed_sim is only called on the all-zarr fusion branch,
    # confirming the concatenated input drove the lazy path.
    assert serialize_calls
    assert list(fused.coords["c"].values) == [0, 1]
    assert result[0, 0].max() == 10
    assert result[0, 1].max() == 30


def test_fuse_concat_t_zarr_backed_sims():
    """Fusing zarr-backed timepoints concatenated along t (chunk-1 axis)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        t0 = _small_zarr_backed_sim(tmpdir, "t0.zarr", 5, t_coord=0.0)
        t1 = _small_zarr_backed_sim(tmpdir, "t1.zarr", 7, t_coord=1.0)

        combined = si_utils.concat([t0, t1], dim="t")
        assert si_utils.is_xarray_zarr_backed(combined)
        assert list(combined.coords["t"].values) == [0.0, 1.0]

        fused = fusion.fuse([combined], transform_key=METADATA_TRANSFORM_KEY)
        result = np.asarray(fused.data)

    assert result.shape[0] == 2
    assert result[0, 0].max() == 5
    assert result[1, 0].max() == 7


def test_fuse_two_zarr_backed_tiles_concat_c_mosaic():
    """A 2-channel concat of two overlapping tiles fuses into one mosaic."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Two tiles offset in x, each concatenated with a second channel.
        tile_a = si_utils.concat(
            [
                _small_zarr_backed_sim(tmpdir, "a0.zarr", 10, tx=0.0, c_coord=0),
                _small_zarr_backed_sim(tmpdir, "a1.zarr", 40, tx=0.0, c_coord=1),
            ],
            dim="c",
        )
        tile_b = si_utils.concat(
            [
                _small_zarr_backed_sim(tmpdir, "b0.zarr", 20, tx=8.0, c_coord=0),
                _small_zarr_backed_sim(tmpdir, "b1.zarr", 50, tx=8.0, c_coord=1),
            ],
            dim="c",
        )
        assert si_utils.is_xarray_zarr_backed(tile_a)
        assert si_utils.is_xarray_zarr_backed(tile_b)

        fused = fusion.fuse(
            [tile_a, tile_b], transform_key=METADATA_TRANSFORM_KEY
        )
        result = np.asarray(fused.data)

    # Mosaic spans both tiles' x-extent; both channels present.
    assert result.shape[1] == 2
    assert result.shape[-1] > 16
    assert result[0, 0].max() == 20
    assert result[0, 1].max() == 50


def test_fuse_msim_concat_c_zarr_backed():
    """Fuse the highest scale of a zarr-backed msim concatenated along c."""
    with tempfile.TemporaryDirectory() as tmpdir:
        m0 = msi_utils.get_msim_from_sim(
            _small_zarr_backed_sim(tmpdir, "m0.zarr", 11, c_coord=0),
            scale_factors=[],
        )
        m1 = msi_utils.get_msim_from_sim(
            _small_zarr_backed_sim(tmpdir, "m1.zarr", 22, c_coord=1),
            scale_factors=[],
        )

        combined = msi_utils.concat([m0, m1], dim="c")
        sim0 = msi_utils.get_sim_from_msim(combined)
        assert si_utils.is_xarray_zarr_backed(sim0)
        assert list(sim0.coords["c"].values) == [0, 1]

        fused = fusion.fuse([sim0], transform_key=METADATA_TRANSFORM_KEY)
        result = np.asarray(fused.data)

    assert result[0, 0].max() == 11
    assert result[0, 1].max() == 22


def test_fuse_msim_concat_c_zarr_backed_multiscale():
    """Concat two multiscale (pyramid) zarr-backed msims along c and fuse.

    Exercises the get_msim_from_sims reassembly across genuine resolution
    levels, all of which must stay zarr-backed.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        def write_read(value, c_coord, name):
            sim = si_utils.get_sim_from_array(
                np.full((1, 1, 256, 256), value, dtype=np.uint16),
                dims=["t", "c", "y", "x"],
                scale={"y": 1.0, "x": 1.0},
                translation={"y": 0.0, "x": 0.0},
                transform_key=METADATA_TRANSFORM_KEY,
                c_coords=[c_coord],
                t_coords=[0.0],
            )
            ngff_utils.write_sim_to_ome_zarr(
                sim,
                os.path.join(tmpdir, name),
                downscale_factors_per_spatial_dim={"y": 2, "x": 2},
                show_progressbar=False,
            )
            return ngff_utils.read_msim_from_ome_zarr(
                os.path.join(tmpdir, name),
                transform_key=METADATA_TRANSFORM_KEY,
            )

        m0 = write_read(11, 0, "m0.zarr")
        m1 = write_read(22, 1, "m1.zarr")

        # More than one scale is required to exercise multiscale reassembly.
        assert len(msi_utils.get_sorted_scale_keys(m0)) > 1

        combined = msi_utils.concat([m0, m1], dim="c")

        for scale_key in msi_utils.get_sorted_scale_keys(combined):
            sim_scale = msi_utils.get_sim_from_msim(combined, scale=scale_key)
            assert si_utils.is_xarray_zarr_backed(sim_scale)
            # Channel labels round-trip through omero metadata as strings.
            assert [str(c) for c in sim_scale.coords["c"].values] == ["0", "1"]

        sim0 = msi_utils.get_sim_from_msim(combined, scale="scale0")
        fused = fusion.fuse([sim0], transform_key=METADATA_TRANSFORM_KEY)
        result = np.asarray(fused.data)

    assert result[0, 0].max() == 11
    assert result[0, 1].max() == 22


def _well_tiles_from_array(arr, tile_size, n_tiles, transform_key):
    """Split one well-sized array into a grid of tiles via spatial isel."""
    well = si_utils.get_sim_from_array(
        arr,
        dims=["c", "y", "x"],
        scale={"y": 1.0, "x": 1.0},
        translation={"y": 0.0, "x": 0.0},
        transform_key=transform_key,
    )
    tiles = []
    for oy in range(0, n_tiles * tile_size, tile_size):
        for ox in range(0, n_tiles * tile_size, tile_size):
            tile = well.isel(
                y=slice(oy, oy + tile_size), x=slice(ox, ox + tile_size)
            )
            tiles.append(
                si_utils.get_sim_from_array(
                    tile,
                    dims=tile.dims,
                    scale={"y": 1.0, "x": 1.0},
                    translation={"y": float(oy), "x": float(ox)},
                    transform_key=transform_key,
                )
            )
    return tiles


@pytest.mark.parametrize(
    "fuse_mode", ["lazy", "output_zarr", "output_zarr_joblib"]
)
def test_fuse_spatial_subviews_of_one_zarr_array(fuse_mode):
    """
    Tiles that are spatial sub-views of one zarr array fuse to the same result
    as their dask-wrapped equivalent.

    Regression test: the all-zarr fusion path used to rebuild every tile from
    its whole backing array, silently producing wrong output.
    """
    data = np.random.default_rng(0).integers(
        1, 1000, size=(1, 200, 200), dtype=np.uint16
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        zarray = zarr.create_array(
            os.path.join(tmpdir, "well.zarr"),
            shape=data.shape,
            chunks=(1, 50, 50),
            dtype=data.dtype,
        )
        zarray[:] = data

        results = {}
        for backend, arr in [("dask", da.from_zarr(zarray)), ("zarr", zarray)]:
            tiles = _well_tiles_from_array(
                arr, tile_size=100, n_tiles=2, transform_key="k"
            )
            assert si_utils.is_xarray_zarr_backed(tiles[0]) == (backend == "zarr")

            fuse_kwargs = {"transform_key": "k", "output_chunksize": 50}
            if fuse_mode == "lazy":
                fused = fusion.fuse(tiles, **fuse_kwargs)
                results[backend] = np.asarray(fused.data).squeeze()
                continue

            if fuse_mode == "output_zarr_joblib":
                fuse_kwargs["batch_options"] = {
                    "batch_func": misc_utils.process_batch_using_joblib,
                    "n_batch": 4,
                }
            output_path = os.path.join(tmpdir, f"fused_{backend}.zarr")
            fusion.fuse(tiles, output_zarr_url=output_path, **fuse_kwargs)
            results[backend] = np.asarray(
                zarr.open_array(output_path, mode="r")[:]
            ).squeeze()

    np.testing.assert_array_equal(results["zarr"], results["dask"])
    np.testing.assert_array_equal(results["zarr"], data.squeeze())


@pytest.mark.parametrize("fuse_mode", ["lazy", "output_zarr"])
def test_fuse_rotated_zarr_backed_tiles_matches_dask(fuse_mode):
    """
    Rotated zarr-backed tiles fuse to the same result as dask-backed ones.

    Regression test: for rotated views the region read from a tile doesn't
    start on its pixel grid, and zarr-backed tiles were resampled with a
    sub-pixel offset.
    """
    data = np.random.default_rng(0).integers(
        1, 1000, size=(1, 200, 200), dtype=np.uint16
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        zarray = zarr.create_array(
            os.path.join(tmpdir, "well.zarr"),
            shape=data.shape,
            chunks=(1, 50, 50),
            dtype=data.dtype,
        )
        zarray[:] = data

        results = {}
        for backend, arr in [("dask", da.from_zarr(zarray)), ("zarr", zarray)]:
            tiles = _well_tiles_from_array(
                arr, tile_size=100, n_tiles=2, transform_key="k"
            )
            for itile, tile in enumerate(tiles):
                angle = 0.01 * (itile + 1)
                affine = np.eye(3)
                affine[:2, :2] = [
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)],
                ]
                si_utils.set_sim_affine(
                    tile,
                    param_utils.affine_to_xaffine(affine),
                    transform_key="k",
                )
            assert si_utils.is_xarray_zarr_backed(tiles[0]) == (backend == "zarr")

            fuse_kwargs = {"transform_key": "k", "output_chunksize": 50}
            if fuse_mode == "lazy":
                fused = fusion.fuse(tiles, **fuse_kwargs)
                results[backend] = np.asarray(fused.data).squeeze()
                continue

            output_path = os.path.join(tmpdir, f"fused_{backend}.zarr")
            fusion.fuse(tiles, output_zarr_url=output_path, **fuse_kwargs)
            results[backend] = np.asarray(
                zarr.open_array(output_path, mode="r")[:]
            ).squeeze()

    assert results["dask"].max() > 0
    np.testing.assert_array_equal(results["zarr"], results["dask"])


def test_first_view_fusion_picks_first_valid_view():
    nan = np.nan
    views = np.array(
        [
            [nan, 1.0, nan],  # view 0
            [2.0, 2.0, nan],  # view 1
        ]
    )

    fused = fusion.first_view_fusion(views)

    # pixel 0: only view 1 covers it; pixel 1: view 0 wins; pixel 2: no view
    np.testing.assert_array_equal(fused, [2.0, 1.0, nan])


@pytest.mark.parametrize("reverse", [False, True])
def test_fuse_first_view_fusion_respects_input_order(reverse):
    # two constant tiles (values 1 and 2) overlapping in x = 6..7
    sims = [
        si_utils.get_sim_from_array(
            da.ones((1, 1, 8, 8), chunks=(1, 1, 4, 4)) * value,
            dims=["c", "t", "y", "x"],
            scale={"y": 1.0, "x": 1.0},
            translation={"y": 0.0, "x": x_origin},
            transform_key=METADATA_TRANSFORM_KEY,
        )
        for value, x_origin in [(1, 0.0), (2, 6.0)]
    ]
    if reverse:
        sims = sims[::-1]

    fused = fusion.fuse(
        sims,
        transform_key=METADATA_TRANSFORM_KEY,
        fusion_func=fusion.first_view_fusion,
        output_chunksize={"y": 4, "x": 4},
    )
    fused_data = np.asarray(fused.data.compute(scheduler="single-threaded"))

    # the overlap is taken entirely from whichever tile comes first
    overlap_value = 2 if reverse else 1
    np.testing.assert_array_equal(fused_data[..., :, :6], 1)
    np.testing.assert_array_equal(fused_data[..., :, 6:8], overlap_value)
    np.testing.assert_array_equal(fused_data[..., :, 8:], 2)
def _make_chunked_fusion_test_sims(transform, tmpdir, input_backend):
    """Deterministic 2x2 tiles (2 channels, 2 time points) for chunked fusion."""

    rng = np.random.default_rng(0)
    transform_key = "k"
    tile_size, overlap, spacing = 24, 6, 0.5
    labels = ["y", "x", "1"]
    t_coords = np.array([0.0, 1.5])

    sims = []
    for itile, (iy, ix) in enumerate(np.ndindex(2, 2)):
        data = rng.integers(
            1, 1000, size=(2, 2, tile_size, tile_size), dtype=np.uint16
        )
        path = os.path.join(tmpdir, f"tile_{itile}.zarr")
        zarray = zarr.create_array(
            path, shape=data.shape, chunks=(1, 1, 16, 16), dtype=data.dtype
        )
        zarray[:] = data

        sim = si_utils.get_sim_from_array(
            zarray if input_backend == "zarr" else da.from_zarr(zarray),
            dims=["t", "c", "y", "x"],
            scale={"y": spacing, "x": spacing},
            translation={
                "y": iy * (tile_size - overlap) * spacing,
                "x": ix * (tile_size - overlap) * spacing,
            },
            transform_key=transform_key,
            c_coords=["ch0", "ch1"],
            t_coords=t_coords,
        )

        if transform == "fractional_time_dependent":
            # Sub-pixel translations which differ between time points.
            affines = np.stack(
                [
                    param_utils.affine_from_translation(
                        [0.11 + 0.3 * itile + 0.2 * it] * 2
                    )
                    for it in range(len(t_coords))
                ]
            )
            si_utils.set_sim_affine(
                sim,
                xr.DataArray(
                    affines,
                    dims=["t", "x_in", "x_out"],
                    coords={"t": t_coords, "x_in": labels, "x_out": labels},
                ),
                transform_key=transform_key,
            )
        elif transform == "rotation":
            angle = 0.05 * (itile + 1)
            affine = np.eye(3)
            affine[:2, :2] = [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
            affine[:2, 2] = 0.37 * itile
            si_utils.set_sim_affine(
                sim,
                param_utils.affine_to_xaffine(affine),
                transform_key=transform_key,
            )

        assert si_utils.is_xarray_zarr_backed(sim) == (input_backend == "zarr")
        sims.append(sim)

    return sims, transform_key


@pytest.mark.parametrize("input_backend", ["dask", "zarr"])
@pytest.mark.parametrize("batch_mode", ["sequential", "joblib"])
@pytest.mark.parametrize(
    "transform", ["translation", "fractional_time_dependent", "rotation"]
)
def test_fuse_to_zarr_matches_chunkwise_lazy_fusion(
    input_backend, batch_mode, transform
):
    """
    Chunked fusion to zarr is bit-identical to fusing each chunk on its own.

    Each chunk of the output must equal the result of a separate (lazy) fusion
    whose output stack is that chunk. The chunked path shares all setup work
    between chunks and only reads the tiles overlapping a chunk, which must not
    change the result.
    """
    output_chunksize = {"y": 16, "x": 20}
    fuse_kwargs = {"output_chunksize": output_chunksize}
    batch_options = None
    if batch_mode == "joblib":
        batch_options = {
            "batch_func": misc_utils.process_batch_using_joblib,
            "n_batch": 5,
            "batch_func_kwargs": {"n_jobs": 2},
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        sims, transform_key = _make_chunked_fusion_test_sims(
            transform, tmpdir, input_backend
        )
        # The reference fuses each chunk on its own, from the same kind of
        # tiles. Zarr-backed tiles are rebuilt from their serialized form, as
        # workers do, before selecting non-spatial coordinates.
        reference_sims = sims
        if input_backend == "zarr":
            reference_sims = [
                si_utils.deserialize_zarr_backed_sim(
                    si_utils.serialize_zarr_backed_sim(sim)
                )
                for sim in sims
            ]
        output_path = os.path.join(tmpdir, "fused.zarr")
        fused = fusion.fuse(
            sims,
            transform_key=transform_key,
            output_zarr_url=output_path,
            batch_options=batch_options,
            **fuse_kwargs,
        )
        result = np.asarray(zarr.open_array(output_path, mode="r")[:])

        osp = si_utils.get_stack_properties_from_sim(fused)
        sdims = si_utils.get_spatial_dims_from_sim(fused)
        nsdims = si_utils.get_nonspatial_dims_from_sim(fused)
        chunk_offsets = {
            dim: range(0, osp["shape"][dim], output_chunksize[dim])
            for dim in sdims
        }
        assert all(len(chunk_offsets[dim]) > 1 for dim in sdims)

        expected = np.zeros_like(result)
        for ns_index in np.ndindex(*[fused.sizes[dim] for dim in nsdims]):
            ns_sims = [
                si_utils.sim_sel_coords(
                    sim,
                    {
                        dim: sim.coords[dim][[i]]
                        for dim, i in zip(nsdims, ns_index)
                    },
                )
                for sim in reference_sims
            ]
            for offsets in itertools.product(
                *[chunk_offsets[dim] for dim in sdims]
            ):
                chunk_shape = {
                    dim: min(output_chunksize[dim], osp["shape"][dim] - offset)
                    for dim, offset in zip(sdims, offsets)
                }
                chunk = fusion.fuse(
                    ns_sims,
                    transform_key=transform_key,
                    output_origin={
                        dim: offset * osp["spacing"][dim] + osp["origin"][dim]
                        for dim, offset in zip(sdims, offsets)
                    },
                    output_shape=chunk_shape,
                    output_spacing=osp["spacing"],
                    **fuse_kwargs,
                ).data.compute(scheduler="single-threaded")
                expected[
                    tuple(slice(i, i + 1) for i in ns_index)
                    + tuple(
                        slice(offset, offset + chunk_shape[dim])
                        for dim, offset in zip(sdims, offsets)
                    )
                ] = chunk

    assert result.max() > 0
    # every time point and channel contains data
    assert all(result[ns_index].max() > 0 for ns_index in np.ndindex(2, 2))
    assert len({result[ns_index].tobytes() for ns_index in np.ndindex(2, 2)}) == 4
    np.testing.assert_array_equal(result, expected)


def test_prepare_block_fusion_shares_setup_between_chunks(monkeypatch):
    """
    Per-chunk work of chunked fusion must not grow with the number of tiles:
    setup is done once, and only tiles overlapping a chunk are read.
    """
    tile_size, n_tiles = 16, 4
    data = np.random.default_rng(0).integers(
        1, 1000, size=(1, n_tiles * tile_size, n_tiles * tile_size)
    ).astype(np.uint16)

    with tempfile.TemporaryDirectory() as tmpdir:
        zarray = zarr.create_array(
            os.path.join(tmpdir, "well.zarr"),
            shape=data.shape,
            chunks=(1, tile_size, tile_size),
            dtype=data.dtype,
        )
        zarray[:] = data
        tiles = _well_tiles_from_array(
            zarray, tile_size=tile_size, n_tiles=n_tiles, transform_key="k"
        )

        calls = {"affines_read": 0, "stack_properties": 0, "tiles_read": []}
        original_get_affine = fusion_core.si_utils.get_affine_from_sim
        original_stack_properties = fusion_core.calc_fusion_stack_properties
        original_fuse_block_views = fusion_core._fuse_block_views

        def count_affines_read(*args, **kwargs):
            calls["affines_read"] += 1
            return original_get_affine(*args, **kwargs)

        def count_stack_properties(*args, **kwargs):
            calls["stack_properties"] += 1
            return original_stack_properties(*args, **kwargs)

        def record_tiles_read(sims_slices, *args, **kwargs):
            calls["tiles_read"].append(len(sims_slices))
            return original_fuse_block_views(sims_slices, *args, **kwargs)

        monkeypatch.setattr(
            fusion_core.si_utils, "get_affine_from_sim", count_affines_read
        )
        monkeypatch.setattr(
            fusion_core, "calc_fusion_stack_properties", count_stack_properties
        )
        monkeypatch.setattr(
            fusion_core, "_fuse_block_views", record_tiles_read
        )

        output_path = os.path.join(tmpdir, "fused.zarr")
        info = fusion.prepare_block_fusion(
            output_path,
            fuse_kwargs={
                "images": tiles,
                "transform_key": "k",
                "output_chunksize": tile_size,
            },
            verbose=False,
        )
        block_ids = list(np.ndindex(*info["nblocks"]))
        assert len(block_ids) == n_tiles**2
        # Chunks are fused from a pickled copy of the plan, as in a worker
        # process.
        fuse_chunk = cloudpickle.loads(cloudpickle.dumps(info["func"]))
        affines_read_during_setup = calls["affines_read"]
        for block_id in block_ids:
            fuse_chunk(block_id)

        result = np.asarray(zarr.open_array(output_path, mode="r")[:])

    np.testing.assert_array_equal(result.squeeze(), data.squeeze())
    assert calls["stack_properties"] == 1
    # affines are read from the tiles once (in setup), not per chunk
    assert affines_read_during_setup == calls["affines_read"]
    # each chunk coincides with exactly one of the 16 tiles
    assert calls["tiles_read"] == [1] * len(block_ids)
