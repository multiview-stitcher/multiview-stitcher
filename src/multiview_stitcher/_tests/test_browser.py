"""
Tests for the browser execution environment.

These run on CPython and cover the platform boundaries the browser depends on:
JSON round trips, reading OME-Zarr through an HTTP-shaped store, the worker
RPC surface, distributed registration through a bridge, cache invalidation and
the virtual OME-Zarr chunks Neuroglancer reads.
"""

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import xarray as xr

from multiview_stitcher import (
    msi_utils,
    ngff_utils,
    registration,
    sample_data,
)
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher.browser import (
    FusionOptions,
    LocalBridge,
    RegistrationOptions,
    Session,
    SessionSpec,
    SourceSpec,
    WorkerRuntime,
    directory_fetch,
    open_http_store,
    serialization,
)
from multiview_stitcher.browser import example_data
from multiview_stitcher.browser import store as browser_store


@pytest.fixture
def tiles_on_disk(tmp_path):
    """Two overlapping 2D tiles written as OME-Zarr v0.4."""
    sims = sample_data.generate_tiled_dataset(
        ndim=2,
        N_c=1,
        N_t=1,
        tile_size=256,
        tiles_x=2,
        tiles_y=1,
        overlap=64,
        zoom=8,
        drift_scale=0,
        shift_scale=8,
    )

    urls = []
    for index, sim in enumerate(sims):
        url = str(tmp_path / f"tile_{index}.ome.zarr")
        msim = msi_utils.get_msim_from_sim(sim, scale_factors=[{"y": 2, "x": 2}])
        ngff_utils.write_sim_to_ome_zarr(
            msi_utils.get_sim_from_msim(msim),
            output_zarr_url=url,
            downscale_factors_per_spatial_dim={"y": 2, "x": 2},
            overwrite=True,
            show_progressbar=False,
        )
        urls.append(url)

    return urls


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_affine_json_round_trip():
    from multiview_stitcher import param_utils

    xaffine = param_utils.affine_to_xaffine(
        np.array([[1.0, 0.2, 3.0], [0.0, 1.0, -4.0], [0.0, 0.0, 1.0]]),
        t_coords=[0, 1],
    )

    payload = json.loads(
        json.dumps(serialization.affine_to_json(xaffine))
    )
    restored = serialization.affine_from_json(payload)

    xr.testing.assert_allclose(xaffine, restored)
    assert restored.dims == xaffine.dims
    assert list(restored.coords["t"].values) == [0, 1]


def test_to_jsonable_survives_json_dumps():
    payload = serialization.to_jsonable(
        {
            "int": np.int64(3),
            "float": np.float32(1.5),
            "array": np.arange(4).reshape(2, 2),
            "nested": [{"a": np.bool_(True)}],
        }
    )
    assert json.loads(json.dumps(payload)) == {
        "int": 3,
        "float": 1.5,
        "array": [[0, 1], [2, 3]],
        "nested": [{"a": True}],
    }


def test_stack_properties_round_trip():
    props = {
        "origin": {"y": 1.5, "x": -2.0},
        "spacing": {"y": 0.5, "x": 0.5},
        "shape": {"y": 10, "x": 12},
    }
    restored = serialization.stack_properties_from_json(
        json.loads(json.dumps(serialization.stack_properties_to_json(props)))
    )
    assert restored == props


# ---------------------------------------------------------------------------
# HTTP-shaped zarr store
# ---------------------------------------------------------------------------


def test_open_msim_through_http_store(tiles_on_disk, tmp_path):
    """The store reads exactly what a direct path read produces."""
    fetch = directory_fetch(tmp_path)
    url = "/" + tiles_on_disk[0].split("/")[-1]

    via_http = ngff_utils.read_msim_from_ome_zarr(
        open_http_store(url, fetch=fetch), array_backend="zarr"
    )
    direct = ngff_utils.read_msim_from_ome_zarr(
        tiles_on_disk[0], array_backend="zarr"
    )

    assert msi_utils.get_sorted_scale_keys(
        via_http
    ) == msi_utils.get_sorted_scale_keys(direct)
    np.testing.assert_array_equal(
        np.asarray(msi_utils.get_sim_from_msim(via_http).data),
        np.asarray(msi_utils.get_sim_from_msim(direct).data),
    )
    assert si_utils.get_spacing_from_sim(
        msi_utils.get_sim_from_msim(via_http)
    ) == si_utils.get_spacing_from_sim(msi_utils.get_sim_from_msim(direct))


def test_http_store_reports_missing_keys(tmp_path):
    store = open_http_store("/nowhere", fetch=directory_fetch(tmp_path))
    assert store.fetch_key(".zgroup") is None


def test_http_store_caches_metadata_but_not_chunks(tiles_on_disk, tmp_path):
    calls = []
    base_fetch = directory_fetch(tmp_path)

    def counting_fetch(url):
        calls.append(url)
        return base_fetch(url)

    store = browser_store.HttpStoreBase(
        "/" + tiles_on_disk[0].split("/")[-1], fetch=counting_fetch
    )

    store.fetch_key(".zattrs")
    store.fetch_key(".zattrs")
    assert len(calls) == 1

    store.fetch_key("0/0.0.0.0")
    store.fetch_key("0/0.0.0.0")
    assert len(calls) == 3

    store.clear_cache()
    store.fetch_key(".zattrs")
    assert len(calls) == 4


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


def test_session_load_and_describe(tiles_on_disk):
    session = Session()
    described = session.load(tiles_on_disk)

    assert described["n_views"] == 2
    assert described["generation"] == 1
    assert si_utils.DEFAULT_TRANSFORM_KEY in described["transform_keys"]

    view = described["views"][0]
    assert view["ndim"] == 2
    assert view["spatial_dims"] == ["y", "x"]
    assert len(view["levels"]) >= 1
    # Metadata only - no image data crosses this boundary.
    assert json.loads(json.dumps(described))


def test_session_register_adds_transform_key(tiles_on_disk):
    session = Session()
    session.load(tiles_on_disk)
    generation_before = session.generation
    views_generation_before = session.views_generation

    result = session.register(
        RegistrationOptions(new_transform_key="registered")
    )

    assert result["transform_key"] == "registered"
    assert "registered" in session.transform_keys()
    assert len(result["params"]) == 2
    # Derived images are retired, because they were computed from the
    # transforms; the views themselves are untouched, so their URLs are not.
    assert session.generation > generation_before
    assert session.views_generation == views_generation_before

    restored = serialization.params_from_json(result["params"])
    assert restored[0].shape[-2:] == (3, 3)


def test_session_copies_selected_transform_key():
    session = Session()
    session.load(example_data.example_sources("tiles-3d"))
    source_key = si_utils.DEFAULT_TRANSFORM_KEY
    source_params = [
        msi_utils.get_transform_from_msim(msim, source_key).copy()
        for msim in session.msims
    ]

    result = session.copy_transform(source_key, "manual")

    assert result["source_transform_key"] == source_key
    assert result["transform_key"] == "manual"
    assert "manual" in session.transform_keys()
    for msim, expected in zip(session.msims, source_params):
        xr.testing.assert_equal(
            msi_utils.get_transform_from_msim(msim, "manual"), expected
        )

    with pytest.raises(ValueError, match="already exists"):
        session.copy_transform(source_key, "manual")


def test_session_persists_neuroglancer_placement_edits():
    session = Session()
    session.load(example_data.example_sources("tiles-3d")[:1])
    source_key = si_utils.DEFAULT_TRANSFORM_KEY
    session.copy_transform(source_key, "manual")
    state = session.neuroglancer_state(transform_key="manual")
    transform = json.loads(
        json.dumps(state["layers"][0]["source"]["transform"])
    )
    output_dims = list(transform["outputDimensions"])
    x_row = output_dims.index("x")
    transform["matrix"][x_row][-1] += 4

    before = msi_utils.get_transform_from_msim(
        session.msims[0], "manual"
    ).copy()
    session.update_neuroglancer_transforms(
        "manual", [{"index": 0, "transform": transform}]
    )
    after = msi_utils.get_transform_from_msim(session.msims[0], "manual")

    spacing = si_utils.get_spacing_from_sim(
        msi_utils.get_sim_from_msim(session.msims[0])
    )
    np.testing.assert_allclose(
        after.sel(x_in="x", x_out="1"),
        before.sel(x_in="x", x_out="1") + 4 * spacing["x"],
    )


def test_session_spec_round_trip_reproduces_transforms(tiles_on_disk):
    session = Session()
    session.load(tiles_on_disk)
    session.register(RegistrationOptions(new_transform_key="registered"))

    spec = session.spec()
    payload = json.loads(json.dumps(spec.to_dict()))
    rebuilt = Session.from_spec(SessionSpec.from_dict(payload))

    assert rebuilt.transform_keys() == session.transform_keys()
    for original, copy in zip(session.msims, rebuilt.msims):
        xr.testing.assert_allclose(
            msi_utils.get_transform_from_msim(original, "registered"),
            msi_utils.get_transform_from_msim(copy, "registered"),
        )


# ---------------------------------------------------------------------------
# Distributed registration
# ---------------------------------------------------------------------------


def _pool_bridge(runtime, max_workers=2):
    """A LocalBridge that runs tasks the way the browser worker pool does."""
    pool = ThreadPoolExecutor(max_workers=max_workers)
    return LocalBridge(
        runner=runtime.run_task,
        map_func=lambda func, items: list(pool.map(func, items)),
    )


def test_distributed_registration_matches_local(tiles_on_disk):
    """Registering through the worker pool gives the same transforms."""
    local_session = Session()
    local_session.load(tiles_on_disk)
    local = local_session.register(
        RegistrationOptions(new_transform_key="registered")
    )

    from multiview_stitcher.browser import executors

    remote_session = Session()
    remote_session.load(tiles_on_disk)
    worker = WorkerRuntime()
    executor = executors.RemotePairwiseExecutor(
        remote_session.spec(), bridge=_pool_bridge(worker)
    )
    remote = remote_session.register(
        RegistrationOptions(new_transform_key="registered"),
        pairwise_executor=executor,
    )

    for local_param, remote_param in zip(local["params"], remote["params"]):
        np.testing.assert_allclose(
            np.asarray(local_param["data"]),
            np.asarray(remote_param["data"]),
            atol=1e-6,
        )


def test_bridge_reports_task_failures(tiles_on_disk):
    from multiview_stitcher.browser.bridge import TaskError

    def failing_runner(task):
        raise ValueError("boom")

    bridge = LocalBridge(runner=failing_runner)
    with pytest.raises(TaskError, match="boom"):
        bridge.dispatch([{"kind": "register_pairs"}])


def test_pairwise_executor_rejects_unknown_registration_func():
    from multiview_stitcher.browser import executors

    with pytest.raises(ValueError, match="cannot be dispatched"):
        executors.serialize_register_kwargs(
            {"pairwise_reg_func": registration.registration_ANTsPy}
        )


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def test_fuse_preview_serves_chunks(tiles_on_disk):
    session = Session()
    session.load(tiles_on_disk)
    session.register(RegistrationOptions(new_transform_key="registered"))

    preview = session.fuse_preview(
        FusionOptions(transform_key="registered")
    )
    route = preview["route"]

    kind, payload = session.serve(route, ".zattrs")
    assert kind == "json"
    multiscales = payload["multiscales"][0]
    assert [axis["name"] for axis in multiscales["axes"]] == preview[
        "metadata"
    ]["dims"]

    kind, zarray = session.serve(route, "0/.zarray")
    assert kind == "json"

    chunk_key = "/".join("0" for _ in zarray["chunks"])
    kind, chunk = session.serve(route, f"0/{chunk_key}")
    assert kind == "bytes"
    assert len(chunk) == int(np.prod(zarray["chunks"])) * np.dtype(
        zarray["dtype"]
    ).itemsize


def test_preview_is_lazy(tiles_on_disk):
    """Registering a preview must not compute any pixels."""
    session = Session()
    session.load(tiles_on_disk)
    preview = session.fuse_preview(FusionOptions())

    virtual_zarr = session._virtual_zarrs[preview["route"]]
    assert hasattr(virtual_zarr.sims[0].data, "compute")


def test_fuse_to_ome_zarr_writes_every_level(tiles_on_disk, tmp_path):
    session = Session()
    session.load(tiles_on_disk)
    session.register(RegistrationOptions(new_transform_key="registered"))

    output_url = str(tmp_path / "fused.ome.zarr")
    options = FusionOptions(
        transform_key="registered", output_zarr_url=output_url
    )

    plan = session.fusion_plan(options)
    assert len(plan["levels"]) > 1, "a pyramid is what makes this worth testing"

    for level in plan["levels"]:
        session.fuse_blocks(
            plan["options"], level["level"], level["block_ids"]
        )
    session.finalize_fusion(plan["options"])

    fused = ngff_utils.read_msim_from_ome_zarr(output_url)
    scale_keys = msi_utils.get_sorted_scale_keys(fused)
    assert len(scale_keys) == len(plan["levels"])

    # Every level must hold data, not just the first.
    for scale_key in scale_keys:
        level_sim = msi_utils.get_sim_from_msim(fused, scale=scale_key)
        assert float(np.asarray(level_sim.data).max()) > 0


def test_parallel_block_writes_match_sequential(tiles_on_disk, tmp_path):
    """Concurrent workers writing one output directory agree with one worker.

    Each block is a distinct chunk file, which is what makes this safe; if
    that ever stopped holding, the two outputs would differ.
    """
    reference_url = str(tmp_path / "sequential.ome.zarr")
    parallel_url = str(tmp_path / "parallel.ome.zarr")

    def build(output_url, n_workers):
        session = Session()
        session.load(tiles_on_disk)
        options = FusionOptions(output_zarr_url=output_url)
        plan = session.fusion_plan(options)

        if n_workers == 1:
            for level in plan["levels"]:
                session.fuse_blocks(
                    plan["options"], level["level"], level["block_ids"]
                )
        else:
            from multiview_stitcher.browser import executors

            worker = WorkerRuntime()
            executor = executors.RemoteFusionExecutor(
                session.spec(),
                bridge=_pool_bridge(worker, max_workers=n_workers),
                n_workers=n_workers,
            )
            written = executor(plan["options"], plan["levels"])
            assert written == plan["n_blocks"]

        session.finalize_fusion(plan["options"])
        return plan

    plan = build(reference_url, 1)
    build(parallel_url, 4)

    reference = ngff_utils.read_msim_from_ome_zarr(reference_url)
    parallel = ngff_utils.read_msim_from_ome_zarr(parallel_url)

    assert msi_utils.get_sorted_scale_keys(
        reference
    ) == msi_utils.get_sorted_scale_keys(parallel)

    for scale_key in msi_utils.get_sorted_scale_keys(reference):
        np.testing.assert_array_equal(
            np.asarray(
                msi_utils.get_sim_from_msim(reference, scale=scale_key).data
            ),
            np.asarray(
                msi_utils.get_sim_from_msim(parallel, scale=scale_key).data
            ),
        )

    assert plan["n_blocks"] > 1


# ---------------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------------


def test_registration_retires_previous_preview_routes(tiles_on_disk):
    session = Session()
    session.load(tiles_on_disk)
    preview = session.fuse_preview(FusionOptions())
    stale_route = preview["route"]

    assert session.serve(stale_route, ".zattrs")[0] == "json"

    session.register(RegistrationOptions(new_transform_key="registered"))

    # The old URL must not answer with data computed before registration -
    # and must say why, so an empty layer is never silent.
    kind, reason = session.serve(stale_route, ".zattrs")
    assert kind == "missing"
    assert "retired generation" in reason

    fresh = session.fuse_preview(FusionOptions(transform_key="registered"))
    assert fresh["route"] != stale_route
    assert session.serve(fresh["route"], ".zattrs")[0] == "json"


def test_worker_rebuilds_preview_for_current_generation(tiles_on_disk):
    """A compute worker serves preview chunks without being told to fuse.

    This is what makes the lazily fused preview render in parallel: the viewer
    asks any worker for a chunk, and the worker reconstructs the same fused
    image from the session spec alone.
    """
    session = Session()
    session.load(tiles_on_disk)
    preview = session.fuse_preview(FusionOptions())

    worker = WorkerRuntime()
    spec = json.loads(json.dumps(session.spec().to_dict()))
    assert spec["preview"] is not None

    # No fuse_preview() call here - the worker only ever sees the spec.
    response = worker.run_task(
        {"kind": "serve", "session": spec, "route": preview["route"], "key": "0/.zarray"}
    )
    assert response["kind"] == "json"
    assert response["payload"] == session.serve(preview["route"], "0/.zarray")[1]

    zarray = response["payload"]
    chunk_key = "/".join("0" for _ in zarray["chunks"])
    response = worker.run_task(
        {
            "kind": "serve",
            "session": spec,
            "route": preview["route"],
            "key": f"0/{chunk_key}",
        }
    )
    assert response["kind"] == "bytes"
    assert response["payload"] == session.serve(
        preview["route"], f"0/{chunk_key}"
    )[1]


def test_worker_does_not_serve_a_retired_route(tiles_on_disk):
    """A spec from before a registration must not answer its old URLs."""
    session = Session()
    session.load(tiles_on_disk)
    preview = session.fuse_preview(FusionOptions())
    stale_spec = json.loads(json.dumps(session.spec().to_dict()))

    session.register(RegistrationOptions(new_transform_key="registered"))
    fresh_spec = json.loads(json.dumps(session.spec().to_dict()))

    worker = WorkerRuntime()
    assert (
        worker.run_task(
            {
                "kind": "serve",
                "session": fresh_spec,
                "route": preview["route"],
                "key": ".zattrs",
            }
        )["kind"]
        == "missing"
    )
    # The stale spec still describes the generation the route belongs to, so it
    # answers - which is why the page always dispatches the current spec.
    assert stale_spec["generation"] < fresh_spec["generation"]
    # The views were not touched, so their URLs survive the registration.
    assert stale_spec["views_generation"] == fresh_spec["views_generation"]


def test_worker_session_cache_is_bounded(tiles_on_disk):
    worker = WorkerRuntime()
    worker.cache_size = 2

    base = Session()
    base.load(tiles_on_disk)
    spec = base.spec()

    for generation in range(4):
        payload = spec.to_dict()
        payload["generation"] = generation
        worker.session_for(payload)

    assert len(worker._session_cache) <= 2


# ---------------------------------------------------------------------------
# Worker RPC surface
# ---------------------------------------------------------------------------


def test_worker_json_command_round_trip(tiles_on_disk):
    from multiview_stitcher.browser import worker as worker_module

    worker_module._runtime = WorkerRuntime()

    response = json.loads(
        worker_module.handle_json(
            "load", json.dumps({"sources": tiles_on_disk})
        )
    )
    assert response["ok"] is True
    assert response["result"]["n_views"] == 2

    response = json.loads(
        worker_module.handle_json(
            "register",
            json.dumps({"options": {}, "distribute": False}),
        )
    )
    assert response["ok"] is True
    assert response["result"]["transform_key"] == "registered"

    worker_module._runtime = None


def test_worker_json_reports_errors():
    from multiview_stitcher.browser import worker as worker_module

    worker_module._runtime = WorkerRuntime()
    response = json.loads(worker_module.handle_json("describe", "{}"))
    assert response["ok"] is False
    assert "No dataset" in response["error"]
    worker_module._runtime = None


def test_serve_route_returns_http_shaped_response(tiles_on_disk):
    from multiview_stitcher.browser import worker as worker_module

    worker_module._runtime = WorkerRuntime()
    worker_module.handle_json("load", json.dumps({"sources": tiles_on_disk}))
    preview = json.loads(
        worker_module.handle_json("fuse_preview", json.dumps({"options": {}}))
    )["result"]

    status, content_type, body = worker_module.serve_route(
        preview["route"], ".zattrs"
    )
    assert status == 200
    assert content_type == "application/json"
    assert json.loads(body.decode())["multiscales"]

    status, _, body = worker_module.serve_route(
        preview["route"], "does/not/exist"
    )
    assert status == 404
    assert b"is not a key" in body

    worker_module._runtime = None


# ---------------------------------------------------------------------------
# Viewer state
# ---------------------------------------------------------------------------


def test_neuroglancer_state_uses_selected_transform_key(tiles_on_disk):
    session = Session()
    session.load(tiles_on_disk)
    # Pretend the tiles arrived through the service worker, which is how the
    # browser addresses a folder the user granted access to.
    for index, source in enumerate(session.sources):
        source.url = f"/app/__mvs__/fs/m1/tile_{index}.ome.zarr"

    session.register(RegistrationOptions(new_transform_key="registered"))

    state = session.neuroglancer_state(
        transform_key="registered", api_base="/app/__mvs__"
    )

    assert len(state["layers"]) == 2
    for layer, source in zip(state["layers"], session.sources):
        # Served natively: the viewer reads the OME-Zarr bytes directly.
        assert layer["source"]["url"] == f"zarr://{source.url}"
        assert "matrix" in layer["source"]["transform"]
    assert json.loads(json.dumps(state))


def test_neuroglancer_state_strips_axes_absent_from_native_source(
    tiles_on_disk,
):
    session = Session()
    session.load(tiles_on_disk[:1])
    sim = msi_utils.get_sim_from_msim(session.msims[0])
    assert tuple(sim.dims) == ("t", "c", "y", "x")

    # Model an OME-Zarr whose array omits the singleton time axis. The
    # in-memory spatial image still has t, while a native viewer URL does not.
    session.msims[0]["scale0/image"].attrs[
        ngff_utils.NGFF_SOURCE_DIMS_ATTR
    ] = ["c", "y", "x"]
    session.sources[0].url = "/app/__mvs__/fs/m1/tile_0.ome.zarr"

    native = session.neuroglancer_state(
        transform_key=si_utils.DEFAULT_TRANSFORM_KEY,
        api_base="/app/__mvs__",
    )["layers"][0]
    native_transform = native["source"]["transform"]
    assert list(native_transform["outputDimensions"]) == ["c'", "y", "x"]
    assert np.asarray(native_transform["matrix"]).shape == (3, 4)

    # A virtual view is generated from the expanded sim and therefore keeps
    # all four in-memory dimensions in both its metadata and transform.
    virtual = session.neuroglancer_state(
        transform_key=si_utils.DEFAULT_TRANSFORM_KEY,
        api_base="/app/__mvs__",
        serve_views="virtual",
    )["layers"][0]
    virtual_transform = virtual["source"]["transform"]
    assert list(virtual_transform["outputDimensions"]) == [
        "t",
        "c'",
        "y",
        "x",
    ]
    assert np.asarray(virtual_transform["matrix"]).shape == (4, 5)


def test_neuroglancer_state_includes_preview_layer(tiles_on_disk):
    session = Session()
    session.load(tiles_on_disk)
    preview = session.fuse_preview(FusionOptions())

    state = session.neuroglancer_state(
        transform_key=si_utils.DEFAULT_TRANSFORM_KEY,
        preview_route=preview["route"],
    )

    names = [layer["name"] for layer in state["layers"]]
    assert "fused" in names
    assert preview["route"] in state["layers"][-1]["source"]["url"]


def test_neuroglancer_state_hides_side_panels_and_uses_dimension_layout():
    session = Session()
    session.load(example_data.example_sources("tiles-3d"))

    state = session.neuroglancer_state()

    assert state["layout"] == "4panel"
    assert state["layerListPanel"] == {"visible": False}
    assert state["selectedLayer"] == {"visible": False}


def test_neuroglancer_state_can_show_every_channel():
    session = Session()
    session.load(example_data.example_sources("tiles-3d")[:1])
    sim = msi_utils.get_sim_from_msim(session.msims[0])
    channel = sim.isel(c=0, drop=True)
    multichannel = xr.concat(
        [channel, channel],
        dim=xr.IndexVariable("c", ["green", "magenta"]),
    )
    session.msims[0] = msi_utils.get_msim_from_sim(
        multichannel, scale_factors=[]
    )

    state = session.neuroglancer_state(show_all_channels=True)

    assert len(state["layers"]) == 2
    assert {layer["localPosition"][0] for layer in state["layers"]} == {0, 1}
    assert any("green" in layer["name"] for layer in state["layers"])
    assert any("magenta" in layer["name"] for layer in state["layers"])


def test_preview_layer_is_hidden_under_another_transform_key(tiles_on_disk):
    """A fused image only means anything in the space it was fused in.

    Shown under a different transform key it would sit where the views are
    not, so it stays loaded - switching back must not refuse it - but hidden.
    """
    session = Session()
    session.load(tiles_on_disk)
    session.register(RegistrationOptions(new_transform_key="registered"))

    preview = session.fuse_preview(FusionOptions(transform_key="registered"))

    def preview_layer(transform_key):
        state = session.neuroglancer_state(
            transform_key=transform_key, preview_route=preview["route"]
        )
        return next(
            layer for layer in state["layers"] if layer["name"] == "fused"
        )

    assert preview_layer("registered")["visible"] is True
    assert preview_layer(si_utils.DEFAULT_TRANSFORM_KEY)["visible"] is False

    # The views themselves exist under every key, so they stay visible.
    state = session.neuroglancer_state(
        transform_key=si_utils.DEFAULT_TRANSFORM_KEY,
        preview_route=preview["route"],
    )
    views = [layer for layer in state["layers"] if layer["name"] != "fused"]
    assert views and all(
        layer.get("visible", True) for layer in views
    )


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


def test_registration_options_reject_unknown_methods():
    with pytest.raises(ValueError, match="pairwise registration function"):
        RegistrationOptions(pairwise_reg_func="does_not_exist")
    with pytest.raises(ValueError, match="groupwise resolution method"):
        RegistrationOptions(groupwise_resolution_method="nope")
    with pytest.raises(ValueError, match="pruning method"):
        RegistrationOptions(pre_registration_pruning_method="nope")


def test_fusion_options_forward_interface_controls():
    options = FusionOptions(
        fusion_func="max",
        output_spacing={"y": 2.0, "x": 2.0},
        blending_widths={"y": 8.0, "x": 8.0},
    )

    kwargs = options.fuse_kwargs()
    assert kwargs["output_spacing"] == {"y": 2.0, "x": 2.0}
    assert kwargs["blending_widths"] == {"y": 8.0, "x": 8.0}


def test_source_spec_name_falls_back_to_url():
    assert SourceSpec(url="/data/tile_3.ome.zarr/").resolved_name() == (
        "tile_3.ome.zarr"
    )
    assert SourceSpec(url="/data/x.zarr", name="Tile A").resolved_name() == (
        "Tile A"
    )


# ---------------------------------------------------------------------------
# Routes shared with the service worker
# ---------------------------------------------------------------------------


def test_route_format_matches_service_worker_fixtures():
    """The checked-in JS fixtures must still describe what Python emits.

    `tests/browser/routes.test.mjs` asserts the service worker parses these
    exact strings; regenerate them with `tests/browser/dump_route_fixtures.py`
    when the route format changes.
    """
    from pathlib import Path

    from multiview_stitcher.browser.session import PREVIEW_NAME

    fixtures_path = (
        Path(__file__).resolve().parents[3]
        / "tests"
        / "browser"
        / "fixtures.json"
    )
    if not fixtures_path.is_file():
        pytest.skip("browser route fixtures are not part of this install")

    fixtures = json.loads(fixtures_path.read_text())

    session = Session(session_id="a1b2c3d4e5f6")
    session.generation = 7
    assert session._route(PREVIEW_NAME) == fixtures["route"]

    for request in fixtures["zarr_requests"]:
        assert request["path"] == f"{request['route']}/{request['key']}"

    # A retired generation must produce a different route, which is what makes
    # stale viewer URLs resolvable to "not found" rather than to old data.
    session.generation = 3
    assert session._route(PREVIEW_NAME) == fixtures["stale_request"]["route"]
    assert session._route(PREVIEW_NAME) != fixtures["route"]


# ---------------------------------------------------------------------------
# Example datasets
# ---------------------------------------------------------------------------


def test_example_generation_is_deterministic():
    """Workers rebuild example tiles independently; they must agree exactly."""
    from multiview_stitcher.browser import example_data

    first = msi_utils.get_sim_from_msim(
        example_data.build_msim("tiles-3d", 2)
    )
    second = msi_utils.get_sim_from_msim(
        example_data.build_msim("tiles-3d", 2)
    )

    np.testing.assert_array_equal(
        np.asarray(first.data), np.asarray(second.data)
    )
    assert si_utils.get_origin_from_sim(first) == si_utils.get_origin_from_sim(
        second
    )


def test_example_dataset_is_3d_and_registrable():
    from multiview_stitcher.browser import example_data

    session = Session()
    described = session.load(example_data.example_sources("tiles-3d"))

    assert described["n_views"] == 4
    assert described["views"][0]["ndim"] == 3
    assert described["views"][0]["spatial_dims"] == ["z", "y", "x"]
    assert len(described["views"][0]["levels"]) == 2
    # Generated data cannot be streamed to the viewer directly.
    assert {view["served"] for view in described["views"]} == {"virtual"}

    result = session.register(
        RegistrationOptions(new_transform_key="registered")
    )
    assert len(result["params"]) == 4

    # The example applies a known in-plane offset per tile, so a registration
    # that found nothing (all-zero shifts) would be a failure.
    shifts = np.array(
        [np.asarray(param["data"])[0][:3, 3] for param in result["params"]]
    )
    assert np.abs(shifts[:, 1:]).max() > 0.1
    assert np.all(np.isfinite(shifts))


def test_example_views_are_served_virtually(tmp_path):
    from multiview_stitcher.browser import example_data

    session = Session(session_id="deadbeef")
    session.load(example_data.example_sources("tiles-3d"))

    route = session.view_route(1)
    kind, zattrs = session.serve(route, ".zattrs")
    assert kind == "json"
    assert [axis["name"] for axis in zattrs["multiscales"][0]["axes"]] == [
        "t",
        "c",
        "z",
        "y",
        "x",
    ]

    kind, zarray = session.serve(route, "0/.zarray")
    assert kind == "json"
    chunk_key = "/".join("0" for _ in zarray["chunks"])
    kind, chunk = session.serve(route, f"0/{chunk_key}")
    assert kind == "bytes"
    assert len(chunk) == int(np.prod(zarray["chunks"])) * np.dtype(
        zarray["dtype"]
    ).itemsize

    # And any compute worker reproduces the identical bytes from the spec.
    worker = WorkerRuntime()
    response = worker.run_task(
        {
            "kind": "serve",
            "session": json.loads(json.dumps(session.spec().to_dict())),
            "route": route,
            "key": f"0/{chunk_key}",
        }
    )
    assert response["payload"] == chunk


# ---------------------------------------------------------------------------
# Viewer URLs
# ---------------------------------------------------------------------------


def test_viewer_urls_live_below_the_service_worker_scope(tiles_on_disk):
    """Every viewer URL must sit inside the prefix the service worker claims.

    The app can be published under a sub-path, where a service worker may only
    intercept URLs below its own directory; a root-relative URL would simply
    not be intercepted and the viewer would show an empty layer.
    """
    api_base = "/multiview-stitcher/main/browser/__mvs__"

    session = Session()
    session.load(tiles_on_disk)
    preview = session.fuse_preview(FusionOptions())

    state = session.neuroglancer_state(
        transform_key=si_utils.DEFAULT_TRANSFORM_KEY,
        base_url="https://example.org",
        api_base=api_base,
        preview_route=preview["route"],
    )

    urls = [layer["source"]["url"] for layer in state["layers"]]
    assert len(urls) == 3

    for url in urls:
        assert url.startswith("zarr://https://example.org/multiview-stitcher/")

    assert urls[-1] == (
        f"zarr://https://example.org{api_base}/zarr/{preview['route']}"
    )


def test_sources_the_viewer_cannot_reach_get_a_virtual_route(tiles_on_disk):
    """A source the viewer cannot fetch is exposed through Python instead."""
    session = Session()
    session.load(tiles_on_disk)

    # A plain filesystem path exists only inside Python.
    assert session.describe()["views"][0]["served"] == "virtual"
    assert "/zarr/" in session.source_url(0, api_base="/app/__mvs__")

    # A service-worker URL is readable by the viewer, so it is used as is.
    session.sources[0].url = "/app/__mvs__/fs/m1/tile_0.ome.zarr"
    assert session.describe()["views"][0]["served"] == "native"


def test_serve_views_virtual_overrides_native_streaming(tiles_on_disk):
    session = Session()
    session.load(tiles_on_disk)
    session.sources[0].url = "/app/__mvs__/fs/m1/tile_0.ome.zarr"

    assert session.source_url(0, api_base="/app/__mvs__") == (
        "/app/__mvs__/fs/m1/tile_0.ome.zarr"
    )
    assert "/zarr/" in session.source_url(
        0, api_base="/app/__mvs__", serve_views="virtual"
    )


# ---------------------------------------------------------------------------
# Adding and clearing views
# ---------------------------------------------------------------------------


def test_loading_more_sources_appends(tiles_on_disk):
    from multiview_stitcher.browser import example_data

    session = Session()
    session.load(tiles_on_disk[:1])
    assert session.describe()["n_views"] == 1

    described = session.add(tiles_on_disk[1:])
    assert described["n_views"] == 2
    assert [source.url for source in session.sources] == list(tiles_on_disk)

    # Appending the same source again is a no-op rather than a duplicate view.
    assert session.add(tiles_on_disk[1:])["n_views"] == 2

    # Mixed inputs are rejected with a readable error, not deep inside a graph.
    with pytest.raises(ValueError, match="same dimensionality"):
        session.add(example_data.example_sources("tiles-3d")[:1])


def test_clear_empties_the_session(tiles_on_disk):
    session = Session()
    session.load(tiles_on_disk)
    generation = session.generation

    described = session.clear()
    assert described["n_views"] == 0
    assert described["transform_keys"] == []
    assert session.is_empty()
    assert session.generation > generation

    # The viewer state must still be renderable so the page can clear the view.
    state = session.neuroglancer_state()
    assert state["layers"] == []

    # And loading afterwards works from the empty state.
    assert session.load(tiles_on_disk)["n_views"] == 2


def test_worker_load_replace_and_append(tiles_on_disk):
    from multiview_stitcher.browser import worker as worker_module

    worker_module._runtime = WorkerRuntime()

    response = json.loads(
        worker_module.handle_json(
            "load", json.dumps({"sources": tiles_on_disk[:1]})
        )
    )
    assert response["result"]["n_views"] == 1

    response = json.loads(
        worker_module.handle_json(
            "load",
            json.dumps({"sources": tiles_on_disk[1:], "replace": False}),
        )
    )
    assert response["result"]["n_views"] == 2

    response = json.loads(
        worker_module.handle_json(
            "load", json.dumps({"sources": tiles_on_disk[:1]})
        )
    )
    assert response["result"]["n_views"] == 1

    response = json.loads(worker_module.handle_json("examples", "{}"))
    assert response["result"]["examples"][0]["name"] == "tiles-3d"

    response = json.loads(worker_module.handle_json("clear", "{}"))
    assert response["result"]["n_views"] == 0

    worker_module._runtime = None


# ---------------------------------------------------------------------------
# The register command as JavaScript calls it
# ---------------------------------------------------------------------------


def _install_pool_bridge(runtime, max_workers=2):
    """Make `get_bridge()` return a pool bridge, as the browser runtime does."""
    from multiview_stitcher.browser import bridge as bridge_module

    previous = bridge_module.get_bridge()
    bridge_module.set_bridge(_pool_bridge(runtime, max_workers=max_workers))
    return previous


@pytest.mark.parametrize("sources_kind", ["ome_zarr", "example"])
def test_register_command_distributed_end_to_end(tiles_on_disk, sources_kind):
    """`register` with distribute=True, exactly as the page issues it.

    This is the whole chain in one test: the JSON command surface, the worker
    pool bridge, the executor's task payload and the compute-worker side. A
    mismatch anywhere in it - for instance the registration channel not
    reaching the workers - surfaces here and nowhere else, because calling
    Session.register() directly never builds the executor.
    """
    from multiview_stitcher.browser import bridge as bridge_module
    from multiview_stitcher.browser import example_data
    from multiview_stitcher.browser import worker as worker_module

    sources = (
        tiles_on_disk
        if sources_kind == "ome_zarr"
        else example_data.example_sources("tiles-3d")
    )

    runtime = WorkerRuntime()
    worker_module._runtime = runtime
    previous = _install_pool_bridge(runtime)

    try:
        response = json.loads(
            worker_module.handle_json(
                "load", json.dumps({"sources": sources})
            )
        )
        assert response["ok"], response

        response = json.loads(
            worker_module.handle_json(
                "register", json.dumps({"options": {}, "distribute": True})
            )
        )
        assert response["ok"], response.get("error")

        params = response["result"]["params"]
        assert len(params) == len(sources)

        ndim = len(json.loads(
            worker_module.handle_json("describe", "{}")
        )["result"]["views"][0]["spatial_dims"])
        for param in params:
            matrix = np.asarray(param["data"])
            assert matrix.shape[-2:] == (ndim + 1, ndim + 1)
            assert np.all(np.isfinite(matrix))
    finally:
        bridge_module.set_bridge(previous)
        worker_module._runtime = None


def test_distributed_registration_matches_local_for_the_example():
    """Distributed and local registration must agree on multi-channel data."""
    from multiview_stitcher.browser import example_data, executors

    sources = example_data.example_sources("tiles-3d")

    local = Session()
    local.load(sources)
    local_result = local.register(
        RegistrationOptions(new_transform_key="registered")
    )

    remote = Session()
    remote.load(sources)
    remote_result = remote.register(
        RegistrationOptions(new_transform_key="registered"),
        pairwise_executor=executors.RemotePairwiseExecutor(
            remote.spec(), bridge=_pool_bridge(WorkerRuntime())
        ),
    )

    for expected, actual in zip(
        local_result["params"], remote_result["params"]
    ):
        np.testing.assert_allclose(
            np.asarray(expected["data"]),
            np.asarray(actual["data"]),
            atol=1e-6,
        )


def test_executor_reads_the_channel_selection_off_the_views(tiles_on_disk):
    """The channel is derived from the views, not passed in separately."""
    from multiview_stitcher.browser import executors

    session = Session()
    session.load(tiles_on_disk)

    assert executors.selected_channel(session.msims[0]) is None

    reduced = session.registration_msims(reg_channel="channel 0")
    assert executors.selected_channel(reduced[0]) == "channel 0"


# ---------------------------------------------------------------------------
# Keeping the view list and the viewer in step
# ---------------------------------------------------------------------------


def test_remove_view(tiles_on_disk):
    session = Session()
    session.load(tiles_on_disk)

    generation = session.generation
    described = session.remove(0)

    assert described["n_views"] == 1
    assert [source.url for source in session.sources] == [tiles_on_disk[1]]
    assert session.generation > generation

    with pytest.raises(IndexError, match="does not exist"):
        session.remove(5)

    assert session.remove(0)["n_views"] == 0
    assert session.is_empty()


def test_failed_append_leaves_the_session_untouched(tiles_on_disk):
    """An incompatible source must not half-load into the session."""
    from multiview_stitcher.browser import example_data

    session = Session()
    session.load(tiles_on_disk)
    before = session.describe()

    with pytest.raises(ValueError, match="same dimensionality"):
        session.add(example_data.example_sources("tiles-3d")[:1])

    after = session.describe()
    assert after["n_views"] == before["n_views"]
    assert [source.url for source in session.sources] == list(tiles_on_disk)
    # And the session is still usable afterwards.
    assert session.register(RegistrationOptions())["params"]


def test_viewer_layers_track_the_view_list(tiles_on_disk):
    """Whatever the app lists, the viewer shows - by name and in order."""
    session = Session()
    session.load(tiles_on_disk)

    def layer_names(described):
        state = session.neuroglancer_state(
            transform_key=si_utils.DEFAULT_TRANSFORM_KEY,
            api_base="/app/__mvs__",
        )
        assert len(state["layers"]) == described["n_views"]
        return [layer["name"] for layer in state["layers"]]

    described = session.describe()
    expected = [
        f"{index}: {view['name']}"
        for index, view in enumerate(described["views"])
    ]
    assert layer_names(described) == expected

    described = session.remove(0)
    expected = [
        f"{index}: {view['name']}"
        for index, view in enumerate(described["views"])
    ]
    assert layer_names(described) == expected
    assert len(expected) == 1


def test_worker_remove_command(tiles_on_disk):
    from multiview_stitcher.browser import worker as worker_module

    worker_module._runtime = WorkerRuntime()
    worker_module.handle_json("load", json.dumps({"sources": tiles_on_disk}))

    response = json.loads(
        worker_module.handle_json("remove", json.dumps({"index": 0}))
    )
    assert response["ok"], response
    assert response["result"]["n_views"] == 1

    response = json.loads(
        worker_module.handle_json("remove", json.dumps({"index": 9}))
    )
    assert response["ok"] is False
    assert "does not exist" in response["error"]

    worker_module._runtime = None


# ---------------------------------------------------------------------------
# A preview must be servable by a worker that predates it
# ---------------------------------------------------------------------------


def test_preview_is_servable_by_a_worker_that_predates_it(tiles_on_disk):
    """The regression behind an empty fused layer in the viewer.

    A compute worker rebuilds a session from the spec and caches it. If a
    preview created afterwards shares that session's generation, the cached
    worker has never heard of it and answers "not found" for every key - which
    zarr renders as an empty image, with no error anywhere.
    """
    session = Session()
    session.load(tiles_on_disk)

    worker = WorkerRuntime()

    # The viewer reads something first, so the worker caches the session.
    before = json.loads(json.dumps(session.spec().to_dict()))
    assert before["preview"] is None
    assert (
        worker.run_task(
            {
                "kind": "serve",
                "session": before,
                "route": session.view_route(0),
                "key": ".zattrs",
            }
        )["kind"]
        == "json"
    )

    # Only now does the user fuse.
    preview = session.fuse_preview(FusionOptions())
    after = json.loads(json.dumps(session.spec().to_dict()))

    for key in (".zattrs", ".zgroup", "0/.zarray"):
        response = worker.run_task(
            {
                "kind": "serve",
                "session": after,
                "route": preview["route"],
                "key": key,
            }
        )
        assert response["kind"] == "json", (key, response)


def test_fusing_retires_the_previous_preview_route(tiles_on_disk):
    """Each fusion gets its own URLs, so a viewer never reads a stale one."""
    session = Session()
    session.load(tiles_on_disk)

    first = session.fuse_preview(FusionOptions())
    second = session.fuse_preview(FusionOptions())

    assert first["route"] != second["route"]
    assert second["generation"] > first["generation"]
    assert session.serve(second["route"], ".zattrs")[0] == "json"
    assert session.serve(first["route"], ".zattrs")[0] == "missing"


def test_missing_routes_explain_themselves(tiles_on_disk):
    """A 404 must say why, otherwise it renders as empty space in silence."""
    from multiview_stitcher.browser import worker as worker_module

    session = Session()
    session.load(tiles_on_disk)
    preview = session.fuse_preview(FusionOptions())

    kind, reason = session.serve("someone-else/g1/fused.ome.zarr", ".zattrs")
    assert kind == "missing"
    assert "retired generation" in reason

    kind, reason = session.serve(preview["route"], "nope/nope")
    assert kind == "missing"
    assert "not a key" in reason

    worker_module._runtime = WorkerRuntime()
    worker_module.handle_json("load", json.dumps({"sources": tiles_on_disk}))
    status, _, body = worker_module.serve_route(
        "someone-else/g1/fused.ome.zarr", ".zattrs"
    )
    assert status == 404
    assert b"retired generation" in body
    worker_module._runtime = None


# ---------------------------------------------------------------------------
# A failed load must not destroy a working session
# ---------------------------------------------------------------------------


def test_failed_load_keeps_the_previous_session_serving(tiles_on_disk):
    """The regression behind a preview that emptied itself.

    Replacing the loaded data used to install the new session before knowing
    it would open. A load that then failed left an empty session behind, and
    every URL the viewer still held - including the fused preview it was
    displaying - answered "not found" from a session that had never loaded
    anything.
    """
    from multiview_stitcher.browser import worker as worker_module

    worker_module._runtime = WorkerRuntime()
    try:
        worker_module.handle_json(
            "load", json.dumps({"sources": tiles_on_disk})
        )
        preview = json.loads(
            worker_module.handle_json(
                "fuse_preview", json.dumps({"options": {}})
            )
        )["result"]

        session_before = worker_module._runtime.session
        status, _, _ = worker_module.serve_route(preview["route"], ".zattrs")
        assert status == 200

        response = json.loads(
            worker_module.handle_json(
                "load",
                json.dumps({"sources": ["/does/not/exist.ome.zarr"]}),
            )
        )
        assert response["ok"] is False

        # The working session is untouched, and still serving.
        assert worker_module._runtime.session is session_before
        status, _, _ = worker_module.serve_route(preview["route"], ".zattrs")
        assert status == 200

        described = json.loads(
            worker_module.handle_json("describe", "{}")
        )["result"]
        assert described["n_views"] == 2
    finally:
        worker_module._runtime = None


def test_failed_load_of_incompatible_views_keeps_the_session(tiles_on_disk):
    from multiview_stitcher.browser import example_data
    from multiview_stitcher.browser import worker as worker_module

    worker_module._runtime = WorkerRuntime()
    try:
        worker_module.handle_json(
            "load", json.dumps({"sources": tiles_on_disk})
        )
        session_before = worker_module._runtime.session

        response = json.loads(
            worker_module.handle_json(
                "load",
                json.dumps(
                    {
                        "sources": example_data.example_sources("tiles-3d"),
                        "replace": False,
                    }
                ),
            )
        )
        assert response["ok"] is False
        assert "same dimensionality" in response["error"]

        assert worker_module._runtime.session is session_before
        assert (
            json.loads(worker_module.handle_json("describe", "{}"))["result"][
                "n_views"
            ]
            == 2
        )
    finally:
        worker_module._runtime = None


def test_rebuilding_from_an_empty_spec_is_an_error():
    """An empty spec must fail loudly rather than answer as a blank session.

    A session built from nothing gets a fresh id at generation 0 and then
    reports every route as retired - a mute 404 in place of a plain bug.
    """
    with pytest.raises(ValueError, match="empty spec"):
        Session.from_spec({})

    with pytest.raises(ValueError, match="empty spec"):
        Session.from_spec({"sources": [], "session_id": "abc", "generation": 3})

    worker = WorkerRuntime()
    with pytest.raises(ValueError, match="empty spec"):
        worker.session_for({})


def test_unusable_spec_falls_back_to_the_local_session(tiles_on_disk):
    """A worker that holds the data answers even if the spec is unusable.

    Pages and wheels can disagree about the spec format across builds. The
    session worker holds the authoritative session, so it should answer from
    it rather than refuse; a compute worker has nothing to fall back on and
    must fail loudly so the page retries elsewhere.
    """
    from multiview_stitcher.browser import worker as worker_module

    worker_module._runtime = WorkerRuntime()
    try:
        worker_module.handle_json(
            "load", json.dumps({"sources": tiles_on_disk})
        )
        preview = json.loads(
            worker_module.handle_json(
                "fuse_preview", json.dumps({"options": {}})
            )
        )["result"]

        # The session worker answers despite the useless spec.
        status, _, body = worker_module.serve_route(
            preview["route"], ".zattrs", {}
        )
        assert status == 200, body

        # A worker without a session of its own reports the failure.
        worker_module._runtime = WorkerRuntime()
        status, _, body = worker_module.serve_route(
            preview["route"], ".zattrs", {}
        )
        assert status == 500
        assert b"empty spec" in body
    finally:
        worker_module._runtime = None


def test_fusion_writes_through_the_service_worker_store(tiles_on_disk, tmp_path):
    """Fusing to disk through the write path the browser actually uses.

    In the browser the output directory is reached over HTTP - a PUT per chunk
    file - rather than as a filesystem path. Exercising that store here is
    what makes the parallel write path testable at all.
    """
    from multiview_stitcher.browser import store as browser_store

    output = tmp_path / "written"
    output.mkdir()

    fetch = browser_store.directory_fetch(output)
    write = browser_store.directory_write(output)

    session = Session(fetch=fetch, write=write)
    session.load(tiles_on_disk)

    options = FusionOptions(output_zarr_url="/__mvs__/fs/out/fused.ome.zarr")
    plan = session.fusion_plan(options)

    for level in plan["levels"]:
        session.fuse_blocks(plan["options"], level["level"], level["block_ids"])
    session.finalize_fusion(plan["options"])

    # Read it back off the real filesystem: the store wrote genuine files.
    written_root = output / "__mvs__" / "fs" / "out" / "fused.ome.zarr"
    assert (written_root / ".zattrs").is_file()

    fused = ngff_utils.read_msim_from_ome_zarr(str(written_root))
    assert len(msi_utils.get_sorted_scale_keys(fused)) == len(plan["levels"])
    assert float(
        np.asarray(msi_utils.get_sim_from_msim(fused).data).max()
    ) > 0


def test_read_only_store_refuses_writes(tmp_path):
    from multiview_stitcher.browser import store as browser_store

    store = browser_store.open_http_store(
        "/x", fetch=browser_store.directory_fetch(tmp_path)
    )
    with pytest.raises(NotImplementedError, match="read-only"):
        store.write_key(".zattrs", b"{}")


# ---------------------------------------------------------------------------
# Dispatching work in batches
# ---------------------------------------------------------------------------


def test_dispatch_sends_work_in_batches():
    """One request must not stay open for a whole fusion.

    A browser terminates a service worker whose event outruns its budget, so a
    single request covering every block is eventually killed mid-flight.
    """
    calls = []

    class RecordingBridge(LocalBridge):
        def call(self, endpoint, payload):
            calls.append(len(payload["tasks"]))
            return super().call(endpoint, payload)

    bridge = RecordingBridge(runner=lambda task: {"n_blocks": 1})
    tasks = [{"kind": "fuse_blocks"} for _ in range(10)]

    results = bridge.dispatch(tasks, batch_size=3)
    assert len(results) == 10
    assert calls == [3, 3, 3, 1]

    # Without a batch size the behaviour is unchanged: one request.
    calls.clear()
    assert len(bridge.dispatch(tasks)) == 10
    assert calls == [10]


def test_dispatch_reports_a_failing_batch_immediately():
    from multiview_stitcher.browser.bridge import TaskError

    seen = []

    def runner(task):
        seen.append(task)
        if len(seen) > 2:
            raise ValueError("boom")
        return {"n_blocks": 1}

    bridge = LocalBridge(runner=runner)
    with pytest.raises(TaskError, match="boom"):
        bridge.dispatch([{"kind": "x"} for _ in range(9)], batch_size=2)

    # Stopped at the failing batch rather than running everything first.
    assert len(seen) < 9


def test_fusion_executor_splits_levels_into_small_tasks(tiles_on_disk):
    from multiview_stitcher.browser import executors

    session = Session()
    session.load(tiles_on_disk)

    dispatched = []

    class CountingBridge(LocalBridge):
        def call(self, endpoint, payload):
            dispatched.append(len(payload["tasks"]))
            return super().call(endpoint, payload)

    worker = WorkerRuntime()
    executor = executors.RemoteFusionExecutor(
        session.spec(),
        bridge=CountingBridge(runner=worker.run_task),
        n_workers=2,
    )

    import tempfile

    with tempfile.TemporaryDirectory() as output:
        options = FusionOptions(
            output_zarr_url=f"{output}/fused.ome.zarr"
        )
        plan = session.fusion_plan(options)
        written = executor(plan["options"], plan["levels"])

    assert written == plan["n_blocks"]
    # Batched: several requests, none of them the whole job.
    assert len(dispatched) > 1
    assert max(dispatched) <= 2


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------


def test_dispatch_reports_progress_per_batch():
    """The page can only learn about progress from the batches it is handed.

    The work itself runs inside one blocking call in the session worker, so
    each dispatch carries how much of the job is already done.
    """
    seen = []

    class ReportingBridge(LocalBridge):
        def call(self, endpoint, payload):
            seen.append(payload.get("progress"))
            return super().call(endpoint, payload)

    bridge = ReportingBridge(runner=lambda task: {"n_blocks": task["units"]})
    tasks = [{"kind": "fuse_blocks", "units": 3} for _ in range(4)]

    bridge.dispatch(
        tasks, batch_size=2, progress={"label": "fusing", "unit": "block"}
    )

    # Each payload also says what its own batch is worth, so the page can
    # finish the bar: the last batch's completion is never reported here.
    assert seen == [
        {"label": "fusing", "unit": "block",
         "completed": 0, "total": 12, "batch": 6},
        {"label": "fusing", "unit": "block",
         "completed": 6, "total": 12, "batch": 6},
    ]
    assert seen[-1]["completed"] + seen[-1]["batch"] == seen[-1]["total"]


def test_progress_counts_blocks_not_tasks(tiles_on_disk, tmp_path):
    """Grouping blocks into tasks must not change what the bar counts."""
    from multiview_stitcher.browser import executors

    reported = []

    class ReportingBridge(LocalBridge):
        def call(self, endpoint, payload):
            if payload.get("progress"):
                reported.append(payload["progress"])
            return super().call(endpoint, payload)

    session = Session()
    session.load(tiles_on_disk)
    options = FusionOptions(
        output_zarr_url=str(tmp_path / "fused.ome.zarr")
    )
    plan = session.fusion_plan(options)

    executor = executors.RemoteFusionExecutor(
        session.spec(),
        bridge=ReportingBridge(runner=WorkerRuntime().run_task),
        n_workers=2,
    )
    executor(plan["options"], plan["levels"])

    assert reported, "a multi-batch job must report progress"
    assert all(item["total"] == plan["n_blocks"] for item in reported)
    assert reported[0]["completed"] == 0
    # Monotonic, and never beyond the total.
    completed = [item["completed"] for item in reported]
    assert completed == sorted(completed)
    assert max(completed) < plan["n_blocks"]


def test_registration_reports_progress_in_pairs(tiles_on_disk):
    from multiview_stitcher.browser import executors

    reported = []

    class ReportingBridge(LocalBridge):
        def call(self, endpoint, payload):
            if payload.get("progress"):
                reported.append(payload["progress"])
            return super().call(endpoint, payload)

    session = Session()
    session.load(tiles_on_disk)
    session.register(
        RegistrationOptions(),
        pairwise_executor=executors.RemotePairwiseExecutor(
            session.spec(), bridge=ReportingBridge(runner=WorkerRuntime().run_task)
        ),
    )

    assert reported
    assert reported[0]["unit"] == "pair"
    assert reported[0]["label"] == "registering"
    assert reported[0]["total"] >= 1


def test_registration_does_not_retire_view_routes(tiles_on_disk):
    """A registration must leave the viewer's layers exactly where they are.

    The result reaches Neuroglancer as a source transform, so not one byte of
    what a view route serves changes. Minting new URLs would force the viewer
    to drop every layer and build it again - losing the shader, its contrast
    range and the layout, and refetching data it already holds.
    """
    session = Session()
    session.load(tiles_on_disk)

    def served():
        out = {}
        for index in range(len(session.sources)):
            route = session.view_route(index)
            kind, attrs = session.serve(route, ".zattrs")
            assert kind == "json"
            out[route] = attrs
            for dataset in attrs["multiscales"][0]["datasets"]:
                path = dataset["path"]
                kind, zarray = session.serve(route, f"{path}/.zarray")
                assert kind == "json"
                out[f"{route}/{path}"] = zarray
                sep = zarray.get("dimension_separator", ".")
                origin = sep.join("0" * len(zarray["shape"]))
                kind, chunk = session.serve(route, f"{path}/{origin}")
                assert kind == "bytes", chunk
                out[f"{route}/{path}/chunk"] = chunk
        return out

    before = served()
    views_generation = session.views_generation

    session.register(RegistrationOptions(new_transform_key="registered"))

    assert session.views_generation == views_generation
    assert served() == before
    assert "registered" in session.transform_keys()

    # Which is what the viewer sees: same layer names, same URLs.
    def layers(key):
        state = session.neuroglancer_state(transform_key=key, api_base="/b")
        return [
            (layer["name"], layer["source"]["url"])
            for layer in state["layers"]
        ]

    assert layers("registered") == layers(si_utils.DEFAULT_TRANSFORM_KEY)


def test_worker_session_cache_tracks_transforms(tiles_on_disk):
    """Registering must invalidate a compute worker's cached session.

    It is no longer the generation that tells a worker its copy is out of
    date, so the transforms themselves have to be part of the cache key -
    otherwise a worker cached before a registration would go on fusing with
    the transforms it was built with.
    """
    runtime = WorkerRuntime()

    session = Session()
    session.load(tiles_on_disk)
    stale = runtime.session_for(session.spec().to_dict())
    assert stale.transform_keys() == [si_utils.DEFAULT_TRANSFORM_KEY]

    session.register(RegistrationOptions(new_transform_key="registered"))
    fresh = runtime.session_for(session.spec().to_dict())

    assert fresh is not stale
    assert "registered" in fresh.transform_keys()

    # Asking again with the same spec still reuses the rebuilt session.
    assert runtime.session_for(session.spec().to_dict()) is fresh


def test_progress_reports_enough_to_reach_the_total(tiles_on_disk, tmp_path):
    """A job small enough for one batch must still be able to show 100%.

    Progress can only travel with a dispatch, so the last batch's completion
    is never reported by Python. Each payload carries what its own batch is
    worth, which is what lets the page finish the bar.
    """
    from multiview_stitcher.browser import executors

    reported = []

    class ReportingBridge(LocalBridge):
        def call(self, endpoint, payload):
            if payload.get("progress"):
                reported.append(payload["progress"])
            return super().call(endpoint, payload)

    session = Session()
    session.load(tiles_on_disk)
    options = FusionOptions(output_zarr_url=str(tmp_path / "fused.ome.zarr"))
    plan = session.fusion_plan(options)

    executor = executors.RemoteFusionExecutor(
        session.spec(),
        bridge=ReportingBridge(runner=WorkerRuntime().run_task),
        n_workers=2,
    )
    executor(plan["options"], plan["levels"])

    assert reported
    for item in reported:
        assert item["batch"] >= 1
        assert item["completed"] + item["batch"] <= item["total"]

    last = reported[-1]
    assert last["completed"] + last["batch"] == last["total"], (
        "the final batch must account for the rest of the work"
    )
