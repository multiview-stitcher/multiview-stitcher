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

    result = session.register(
        RegistrationOptions(new_transform_key="registered")
    )

    assert result["transform_key"] == "registered"
    assert "registered" in session.transform_keys()
    assert len(result["params"]) == 2
    assert session.generation > generation_before

    restored = serialization.params_from_json(result["params"])
    assert restored[0].shape[-2:] == (3, 3)


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
        remote_session.spec(),
        bridge=_pool_bridge(worker),
        reg_channel_index=0,
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


def test_fuse_to_ome_zarr_on_disk(tiles_on_disk, tmp_path):
    session = Session()
    session.load(tiles_on_disk)
    session.register(RegistrationOptions(new_transform_key="registered"))

    output_url = str(tmp_path / "fused.ome.zarr")
    options = FusionOptions(
        transform_key="registered", output_zarr_url=output_url
    )

    plan = session.fusion_plan(options)
    assert plan["block_ids"]
    session.fuse_blocks(plan["options"], plan["block_ids"])
    session.finalize_fusion(
        plan["options"], plan["output_stack_properties"]
    )

    fused = ngff_utils.read_msim_from_ome_zarr(output_url)
    fused_sim = msi_utils.get_sim_from_msim(fused)
    expected_shape = plan["output_stack_properties"]["shape"]
    for dim, size in expected_shape.items():
        assert fused_sim.sizes[dim] == size
    assert float(np.asarray(fused_sim.data).max()) > 0


def test_fuse_blocks_split_over_pool_matches_single_worker(
    tiles_on_disk, tmp_path
):
    """Blocks fused by several workers into one store match a single writer."""
    reference_url = str(tmp_path / "reference.ome.zarr")
    split_url = str(tmp_path / "split.ome.zarr")

    def run(output_url, n_parts):
        session = Session()
        session.load(tiles_on_disk)
        options = FusionOptions(output_zarr_url=output_url)
        plan = session.fusion_plan(options)

        from multiview_stitcher.browser.executors import split_evenly

        for group in split_evenly(plan["block_ids"], n_parts):
            # Each group stands in for one worker attaching to the store
            # created by fusion_plan().
            worker_session = Session.from_spec(session.spec())
            worker_session.fuse_blocks(plan["options"], group)

        session.finalize_fusion(
            plan["options"], plan["output_stack_properties"]
        )
        return output_url

    run(reference_url, 1)
    run(split_url, 3)

    reference = msi_utils.get_sim_from_msim(
        ngff_utils.read_msim_from_ome_zarr(reference_url)
    )
    split = msi_utils.get_sim_from_msim(
        ngff_utils.read_msim_from_ome_zarr(split_url)
    )
    np.testing.assert_array_equal(
        np.asarray(reference.data), np.asarray(split.data)
    )


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

    # The old URL must not answer with data computed before registration.
    assert session.serve(stale_route, ".zattrs") == ("missing", None)

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
    assert body is None

    worker_module._runtime = None


# ---------------------------------------------------------------------------
# Viewer state
# ---------------------------------------------------------------------------


def test_neuroglancer_state_uses_selected_transform_key(tiles_on_disk):
    session = Session()
    session.load(tiles_on_disk)
    session.register(RegistrationOptions(new_transform_key="registered"))

    state = session.neuroglancer_state(transform_key="registered")

    assert len(state["layers"]) == 2
    for layer, source in zip(state["layers"], session.sources):
        assert layer["source"]["url"].endswith(source.url)
        assert "matrix" in layer["source"]["transform"]
    assert json.loads(json.dumps(state))


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
