"""
End-to-end smoke test of the browser runtime, executed inside Pyodide.

It walks the whole path the browser app takes - write a miniature multiscale
OME-Zarr, open it through the browser session, register two views, fuse them
lazily and read one fused chunk - and checks the results, so that platform
differences between CPython and Pyodide surface in CI rather than in the UI.

The library layer is no longer one of those differences: the browser runs the
same zarr v3 and the same ngff-zarr as CPython. What is left to catch is the
*runtime* - a numpy or xarray build that behaves differently, a codec that is
not compiled in, and above all a zarr that cannot block here.

Needs a JavaScript runtime with WebAssembly stack switching (JSPI): zarr v3
has no thread to run an event loop on in the browser and suspends instead.
Node.js 25 and later have it unflagged; 20 to 24 need
``--experimental-wasm-jspi``.

Run by ``tests/browser/smoke.mjs``; ``main()`` returns a JSON string.
"""

import json
import sys

import numpy as np

RESULTS = {}


def check(name, condition, detail=""):
    RESULTS[name] = {"ok": bool(condition), "detail": str(detail)}
    if not condition:
        raise AssertionError(f"{name} failed: {detail}")


def build_dataset(root="/data", n_timepoints=1, tile_size=256, prefix="tile"):
    """Write two overlapping tiles as multiscale OME-Zarr v0.4."""
    import os

    from multiview_stitcher import msi_utils, ngff_utils, sample_data

    os.makedirs(root, exist_ok=True)

    sims = sample_data.generate_tiled_dataset(
        ndim=2,
        N_c=1,
        N_t=n_timepoints,
        tile_size=tile_size,
        tiles_x=2,
        tiles_y=1,
        overlap=64,
        zoom=8,
        drift_scale=0,
        shift_scale=8,
    )

    urls = []
    for index, sim in enumerate(sims):
        url = f"{root}/{prefix}_{index}.ome.zarr"
        msim = msi_utils.get_msim_from_sim(
            sim, scale_factors=[{"y": 2, "x": 2}]
        )
        ngff_utils.write_sim_to_ome_zarr(
            msi_utils.get_sim_from_msim(msim),
            output_zarr_url=url,
            downscale_factors_per_spatial_dim={"y": 2, "x": 2},
            overwrite=True,
            show_progressbar=False,
        )
        urls.append(url)

    return urls


def main():
    import zarr
    import zarr.abc.store

    from multiview_stitcher import ngff_utils
    from multiview_stitcher.browser import (
        FusionOptions,
        RegistrationOptions,
        Session,
        WorkerRuntime,
        runtime_info,
    )

    info = runtime_info()
    check("runtime_is_pyodide", info["pyodide"], info)
    check(
        "zarr_v3",
        zarr.__version__.startswith("3."),
        f"zarr {zarr.__version__}",
    )
    # The check that would have caught shipping PyPI's zarr, whose synchronous
    # API starts a thread that Pyodide cannot give it.
    check("zarr_sync_usable", info["zarr_sync"] == "ok", info["zarr_sync"])
    # The same NGFF library as on CPython. A second implementation for the
    # browser was exactly the kind of platform boundary this test exists to
    # catch, so its absence is worth asserting.
    check(
        "ngff_zarr_present",
        ngff_utils.ngff_zarr is not None,
        "ngff-zarr is not importable",
    )

    # --- write and read a miniature multiscale OME-Zarr ------------------
    urls = build_dataset()
    check("dataset_written", len(urls) == 2, urls)

    root = zarr.open_group(urls[0], mode="r")
    multiscales = root.attrs["multiscales"][0]
    check(
        "ome_zarr_v04_metadata",
        multiscales.get("version") == "0.4"
        and len(multiscales["datasets"]) == 2,
        multiscales,
    )

    session = Session()
    described = session.load(urls)
    check("session_loaded", described["n_views"] == 2, described["n_views"])
    check(
        "multiscale_levels",
        len(described["views"][0]["levels"]) == 2,
        described["views"][0]["levels"],
    )
    check(
        "json_serialisable",
        json.loads(json.dumps(described))["n_views"] == 2,
        "describe() must survive the JS boundary",
    )

    # Reading must stay lazy: zarr-backed, not materialised on load.
    from multiview_stitcher import msi_utils
    from multiview_stitcher import spatial_image_utils as si_utils

    sim0 = msi_utils.get_sim_from_msim(session.msims[0])
    check(
        "lazy_input",
        not isinstance(si_utils._get_backend_data(sim0), np.ndarray),
        type(si_utils._get_backend_data(sim0)).__name__,
    )

    # --- register the two views -----------------------------------------
    registered = session.register(
        RegistrationOptions(new_transform_key="registered")
    )
    check(
        "registration_transform_key",
        registered["transform_key"] == "registered"
        and "registered" in session.transform_keys(),
        session.transform_keys(),
    )
    check(
        "registration_params",
        len(registered["params"]) == 2,
        len(registered["params"]),
    )

    params = np.asarray(registered["params"][1]["data"])
    check("registration_param_shape", params.shape[-2:] == (3, 3), params.shape)
    shift = params[0][:2, 2] if params.ndim == 3 else params[:2, 2]
    check(
        "registration_shift_is_finite",
        np.all(np.isfinite(shift)),
        shift.tolist(),
    )

    # --- distributed registration through the worker pool ---------------
    from multiview_stitcher.browser import LocalBridge, executors

    pool_session = Session()
    pool_session.load(urls)
    pool_worker = WorkerRuntime()
    pool_result = pool_session.register(
        RegistrationOptions(new_transform_key="registered"),
        pairwise_executor=executors.RemotePairwiseExecutor(
            pool_session.spec(),
            bridge=LocalBridge(runner=pool_worker.run_task),
        ),
    )
    np.testing.assert_allclose(
        np.asarray(registered["params"][1]["data"]),
        np.asarray(pool_result["params"][1]["data"]),
        atol=1e-6,
    )
    RESULTS["distributed_registration_matches"] = {
        "ok": True,
        "detail": "identical transforms",
    }

    # --- a timelapse, one timepoint per task ----------------------------
    # Each timepoint of a pair is registered on its own and the results are
    # joined back into one array over time. That join is an xarray concat on
    # coordinates that have been through JSON, which is exactly the sort of
    # thing an older array stack in the browser gets subtly wrong.
    time_urls = build_dataset(
        n_timepoints=3, tile_size=128, prefix="timelapse"
    )
    time_local = Session()
    time_local.load(time_urls)
    time_expected = time_local.register(
        RegistrationOptions(new_transform_key="registered")
    )

    dispatched = []

    class RecordingBridge(LocalBridge):
        def call(self, endpoint, payload):
            dispatched.extend(payload["tasks"])
            return super().call(endpoint, payload)

    time_pool = Session()
    time_pool.load(time_urls)
    time_actual = time_pool.register(
        RegistrationOptions(new_transform_key="registered"),
        pairwise_executor=executors.RemotePairwiseExecutor(
            time_pool.spec(),
            bridge=RecordingBridge(runner=WorkerRuntime().run_task),
        ),
    )

    check(
        "timelapse_dispatched_per_timepoint",
        dispatched
        and all(len(task["time_indices"]) == 1 for task in dispatched)
        and sorted({task["time_indices"][0] for task in dispatched})
        == [0, 1, 2],
        [task["time_indices"] for task in dispatched],
    )
    np.testing.assert_allclose(
        np.asarray(time_expected["params"][1]["data"]),
        np.asarray(time_actual["params"][1]["data"]),
        atol=1e-6,
    )
    check(
        "timelapse_registration_matches",
        time_actual["params"][1]["coords"]["t"]
        == time_expected["params"][1]["coords"]["t"],
        time_actual["params"][1]["coords"],
    )

    # --- fuse lazily and read one chunk ---------------------------------
    preview = session.fuse_preview(
        FusionOptions(transform_key="registered")
    )
    route = preview["route"]

    kind, zattrs = session.serve(route, ".zattrs")
    check("preview_zattrs", kind == "json" and "multiscales" in zattrs, kind)

    kind, zarray = session.serve(route, "0/.zarray")
    check("preview_zarray", kind == "json", kind)

    chunks = zarray["chunks"]
    dtype = np.dtype(zarray["dtype"])
    chunk_key = "/".join("0" for _ in chunks)
    kind, chunk = session.serve(route, f"0/{chunk_key}")

    expected_bytes = int(np.prod(chunks)) * dtype.itemsize
    check(
        "fused_chunk_served",
        kind == "bytes" and len(chunk) == expected_bytes,
        f"kind={kind} len={len(chunk) if chunk else None} "
        f"expected={expected_bytes}",
    )

    array = np.frombuffer(chunk, dtype=dtype).reshape(chunks)
    check(
        "fused_chunk_shape",
        list(array.shape) == list(chunks),
        array.shape,
    )
    check(
        "fused_chunk_has_signal",
        float(array.max()) > 0,
        float(array.max()),
    )

    # A compute worker must serve the same chunk from the session spec alone -
    # this is what lets the preview render in parallel across the pool.
    serve_worker = WorkerRuntime()
    served = serve_worker.run_task(
        {
            "kind": "serve",
            "session": json.loads(json.dumps(session.spec().to_dict())),
            "route": route,
            "key": f"0/{chunk_key}",
        }
    )
    check(
        "worker_serves_preview_chunk",
        served["kind"] == "bytes" and served["payload"] == chunk,
        served["kind"],
    )

    # The fused image must cover both tiles.
    fused_shape = preview["metadata"]["levels"][0]["shape"]
    tile_shape = described["views"][0]["levels"][0]["shape"]
    check(
        "fused_covers_both_tiles",
        fused_shape["x"] > tile_shape["x"],
        f"fused={fused_shape} tile={tile_shape}",
    )

    # --- stale routes must not serve data -------------------------------
    session.register(RegistrationOptions(new_transform_key="registered2"))
    stale_kind, stale_reason = session.serve(route, ".zattrs")
    check(
        "stale_route_invalidated",
        stale_kind == "missing" and "retired generation" in str(stale_reason),
        f"{stale_kind}: {stale_reason}",
    )

    # --- fuse to an OME-Zarr on disk, in parallel ------------------------
    # The browser writes each chunk as its own file through the service
    # worker, so several workers can write one output directory at once.
    import os

    from multiview_stitcher.browser import Session as HttpSession

    def _sw_relative(url):
        marker = "/__mvs__/fs/"
        index = url.index(marker) + len(marker)
        return url[index:].split("/", 1)[1]

    def service_worker_fetch(url):
        """Stand in for the service worker reading a granted directory."""
        path = f"/out/{_sw_relative(url)}" if "/fs/out/" in url else (
            f"/data/{_sw_relative(url)}"
        )
        try:
            with open(path, "rb") as handle:
                return handle.read()
        except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
            return None

    def service_worker_write(url, data):
        """Stand in for the fs worker: one file created, written and closed."""
        path = f"/out/{_sw_relative(url)}"

        if data is None:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(bytes(data))

    os.makedirs("/out", exist_ok=True)

    write_session = HttpSession(
        session_id="writer",
        fetch=service_worker_fetch,
        write=service_worker_write,
    )
    write_session.load(
        [
            {"url": f"/browser/__mvs__/fs/m1/tile_{index}.ome.zarr"}
            for index in range(2)
        ]
    )
    disk_options = FusionOptions(
        output_zarr_url="/browser/__mvs__/fs/out/fused.ome.zarr"
    )
    plan = write_session.fusion_plan(disk_options)
    check(
        "fusion_plan_is_multiscale",
        len(plan["levels"]) > 1 and plan["n_blocks"] > 1,
        f"{len(plan['levels'])} level(s), {plan['n_blocks']} block(s)",
    )

    # Split every level's blocks over several "workers", each rebuilding the
    # session from the spec exactly as a compute worker does.
    from multiview_stitcher.browser.executors import split_evenly

    write_spec = json.loads(json.dumps(write_session.spec().to_dict()))
    for level in plan["levels"]:
        for group in split_evenly(level["block_ids"], 3):
            worker_session = HttpSession.from_spec(
                write_spec,
                fetch=service_worker_fetch,
                write=service_worker_write,
            )
            worker_session.fuse_blocks(
                plan["options"], level["level"], group
            )

    write_session.finalize_fusion(plan["options"])

    written = ngff_utils.read_msim_from_ome_zarr("/out/fused.ome.zarr")
    written_levels = msi_utils.get_sorted_scale_keys(written)
    check(
        "parallel_write_produced_every_level",
        len(written_levels) == len(plan["levels"]),
        f"{written_levels} vs {[lvl['path'] for lvl in plan['levels']]}",
    )

    level_maxima = [
        float(
            np.asarray(
                msi_utils.get_sim_from_msim(written, scale=scale_key).data
            ).max()
        )
        for scale_key in written_levels
    ]
    check(
        "parallel_write_levels_have_signal",
        all(value > 0 for value in level_maxima),
        level_maxima,
    )

    # --- OME-Zarr read through the HTTP store, then fused ----------------
    # The browser reads inputs over HTTP, not from a path, and a fused preview
    # of them spans several chunks (unlike the single-chunk example). Both
    # differences only bite under zarr v2, so they are checked here.
    from multiview_stitcher.browser import Session as HttpSession
    from multiview_stitcher.browser import store as browser_store

    def service_worker_fetch(url):
        """Stand in for the service worker: /__mvs__/fs/<mount>/<path>."""
        marker = "/__mvs__/fs/"
        index = url.index(marker) + len(marker)
        relative = url[index:].split("/", 1)[1]
        path = f"/data/{relative}"
        try:
            with open(path, "rb") as handle:
                return handle.read()
        except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
            return None

    http_session = HttpSession(session_id="httpsession", fetch=service_worker_fetch)
    http_described = http_session.load(
        [
            {"url": f"/browser/__mvs__/fs/m1/tile_{index}.ome.zarr"}
            for index in range(2)
        ]
    )
    check(
        "http_store_loaded",
        http_described["n_views"] == 2
        and {view["served"] for view in http_described["views"]} == {"native"},
        http_described["views"][0]["served"],
    )
    check(
        "http_store_is_a_zarr_store",
        isinstance(
            browser_store.open_http_store("/x", fetch=service_worker_fetch),
            zarr.abc.store.Store,
        ),
        "expected a zarr v3 store in Pyodide",
    )

    http_session.register(RegistrationOptions(new_transform_key="registered"))
    http_preview = http_session.fuse_preview(
        FusionOptions(transform_key="registered")
    )

    kind, http_zattrs = http_session.serve(http_preview["route"], ".zattrs")
    http_levels = [
        dataset["path"]
        for dataset in http_zattrs["multiscales"][0]["datasets"]
    ]
    check(
        "http_preview_is_multiscale",
        kind == "json" and len(http_levels) > 1,
        http_levels,
    )

    # Neuroglancer renders the coarsest level first, so a level that fails to
    # serve leaves the layer blank even when level 0 is perfectly fine.
    level_reports = []
    for level in http_levels:
        level_kind, level_zarray = http_session.serve(
            http_preview["route"], f"{level}/.zarray"
        )
        if level_kind != "json":
            level_reports.append((level, "no .zarray", None))
            continue

        level_grid = [
            int(np.ceil(size / chunk))
            for size, chunk in zip(
                level_zarray["shape"], level_zarray["chunks"]
            )
        ]
        level_expected = int(np.prod(level_zarray["chunks"])) * np.dtype(
            level_zarray["dtype"]
        ).itemsize

        bad = []
        for index in np.ndindex(*level_grid):
            chunk_kind, chunk = http_session.serve(
                http_preview["route"],
                f"{level}/" + "/".join(str(i) for i in index),
            )
            if chunk_kind != "bytes" or len(chunk) != level_expected:
                bad.append((index, chunk_kind))
        level_reports.append((level, level_grid, bad))

    check(
        "http_preview_every_level_serves",
        all(
            isinstance(report[1], list) and not report[2]
            for report in level_reports
        ),
        level_reports,
    )

    kind, http_zarray = http_session.serve(http_preview["route"], "0/.zarray")
    check("http_preview_zarray", kind == "json", kind)

    grid = [
        int(np.ceil(size / chunk))
        for size, chunk in zip(http_zarray["shape"], http_zarray["chunks"])
    ]
    check(
        "http_preview_is_multi_chunk",
        int(np.prod(grid)) > 1,
        f"grid={grid} - a single-chunk preview would not exercise this",
    )

    # Every chunk of the grid must serve, not just the first one.
    served_chunks = []
    for index in np.ndindex(*grid):
        kind, chunk = http_session.serve(
            http_preview["route"],
            "0/" + "/".join(str(i) for i in index),
        )
        served_chunks.append((kind, len(chunk) if chunk else 0))

    expected_bytes = int(np.prod(http_zarray["chunks"])) * np.dtype(
        http_zarray["dtype"]
    ).itemsize
    check(
        "http_preview_all_chunks_served",
        all(
            kind == "bytes" and size == expected_bytes
            for kind, size in served_chunks
        ),
        f"{served_chunks} expected {len(served_chunks)}x{expected_bytes}",
    )

    totals = []
    for index in np.ndindex(*grid):
        _, chunk = http_session.serve(
            http_preview["route"],
            "0/" + "/".join(str(i) for i in index),
        )
        totals.append(
            float(
                np.frombuffer(chunk, dtype=np.dtype(http_zarray["dtype"])).max()
            )
        )
    check(
        "http_preview_chunks_have_signal",
        sum(value > 0 for value in totals) >= len(totals) - 1,
        totals,
    )

    # A compute worker must serve these chunks too. It rebuilds the session
    # from the spec, which means re-opening the OME-Zarr over HTTP and
    # reconstructing the fused image - the exact combination that the
    # single-chunk, no-IO example never exercises.
    http_worker = WorkerRuntime(fetch=service_worker_fetch)
    http_spec = json.loads(json.dumps(http_session.spec().to_dict()))
    check(
        "http_spec_carries_preview",
        http_spec["preview"] is not None,
        "a worker cannot rebuild a preview it was never told about",
    )

    worker_reports = []
    for level in http_levels:
        level_response = http_worker.run_task(
            {
                "kind": "serve",
                "session": http_spec,
                "route": http_preview["route"],
                "key": f"{level}/.zarray",
            }
        )
        if level_response["kind"] != "json":
            worker_reports.append((level, level_response["kind"], None))
            continue

        level_zarray = level_response["payload"]
        level_grid = [
            int(np.ceil(size / chunk))
            for size, chunk in zip(
                level_zarray["shape"], level_zarray["chunks"]
            )
        ]
        bad = []
        for index in np.ndindex(*level_grid):
            chunk_response = http_worker.run_task(
                {
                    "kind": "serve",
                    "session": http_spec,
                    "route": http_preview["route"],
                    "key": f"{level}/" + "/".join(str(i) for i in index),
                }
            )
            if chunk_response["kind"] != "bytes":
                bad.append((index, chunk_response["kind"]))
        worker_reports.append((level, level_grid, bad))

    check(
        "http_preview_served_by_compute_worker",
        all(
            isinstance(report[1], list) and not report[2]
            for report in worker_reports
        ),
        worker_reports,
    )

    # A worker that cached the session before the preview existed must still
    # serve it: this failed as an empty layer with every key 404, and nothing
    # in any log.
    stale_worker = WorkerRuntime(fetch=service_worker_fetch)
    stale_session = HttpSession(session_id="stale", fetch=service_worker_fetch)
    stale_session.load(
        [
            {"url": f"/browser/__mvs__/fs/m1/tile_{index}.ome.zarr"}
            for index in range(2)
        ]
    )
    before_spec = json.loads(json.dumps(stale_session.spec().to_dict()))
    stale_worker.run_task(
        {
            "kind": "serve",
            "session": before_spec,
            "route": stale_session.view_route(0),
            "key": ".zattrs",
        }
    )

    stale_preview = stale_session.fuse_preview(FusionOptions())
    after_spec = json.loads(json.dumps(stale_session.spec().to_dict()))
    stale_response = stale_worker.run_task(
        {
            "kind": "serve",
            "session": after_spec,
            "route": stale_preview["route"],
            "key": ".zattrs",
        }
    )
    check(
        "preview_servable_by_worker_that_predates_it",
        stale_response["kind"] == "json",
        stale_response,
    )

    # --- the generated 3D example, served virtually ----------------------
    # 3D registration takes a different path through the overlap graph than
    # 2D, including one that reaches for a process pool that cannot exist in
    # WebAssembly, so it is worth exercising here.
    from multiview_stitcher.browser import example_data

    example = Session()
    described_example = example.load(
        example_data.example_sources("tiles-3d")
    )
    check(
        "example_loaded",
        described_example["n_views"] == 4
        and described_example["views"][0]["ndim"] == 3,
        described_example["views"][0]["levels"][0]["shape"],
    )
    check(
        "example_served_virtually",
        {view["served"] for view in described_example["views"]} == {"virtual"},
        {view["served"] for view in described_example["views"]},
    )

    example_route = example.view_route(0)
    kind, example_zarray = example.serve(example_route, "0/.zarray")
    check("example_view_zarray", kind == "json", kind)

    example_key = "/".join("0" for _ in example_zarray["chunks"])
    kind, example_chunk = example.serve(example_route, f"0/{example_key}")
    check(
        "example_view_chunk",
        kind == "bytes"
        and len(example_chunk)
        == int(np.prod(example_zarray["chunks"]))
        * np.dtype(example_zarray["dtype"]).itemsize,
        kind,
    )

    example_result = example.register(
        RegistrationOptions(new_transform_key="registered")
    )
    example_shifts = np.array(
        [
            np.asarray(param["data"])[0][:3, 3]
            for param in example_result["params"]
        ]
    )
    check(
        "example_registered_3d",
        np.all(np.isfinite(example_shifts))
        and np.abs(example_shifts[:, 1:]).max() > 0.1,
        example_shifts.round(2).tolist(),
    )

    example_state = example.neuroglancer_state(
        transform_key="registered",
        base_url="https://example.org",
        api_base="/browser/__mvs__",
    )
    check(
        "example_viewer_urls_in_sw_scope",
        all(
            layer["source"]["url"].startswith(
                "zarr://https://example.org/browser/__mvs__/"
            )
            for layer in example_state["layers"]
        ),
        [layer["source"]["url"] for layer in example_state["layers"]],
    )

    # --- a mosaic CZI, read by the same reader CPython uses --------------
    # czifile and tifffile are pure Python, but imagecodecs - which czifile
    # reaches for - is a C extension with no WebAssembly build. What is checked
    # here is that its absence leaves czifile usable rather than disabling CZI
    # support altogether, and that the whole browser path works on top of the
    # unchanged `io.read_mosaic_into_sims_czifile`.
    #
    # The sample CZI ships inside the wheel, so this needs no mounted file. In
    # the app the user's own file is mounted through WORKERFS and reaches the
    # same reader as an ordinary path.
    from multiview_stitcher import czi_utils, sample_data
    from multiview_stitcher.browser import czi as browser_czi

    check(
        "czifile_importable_without_imagecodecs",
        czi_utils.czifile is not None,
        "czifile did not import; CZI support would be silently unavailable",
    )
    check(
        "imagecodecs_absent",
        __import__("multiview_stitcher.czifile_patch", fromlist=["x"]).imagecodecs
        is None,
        "imagecodecs is unexpectedly present, so its absence is untested here",
    )

    czi_path = str(sample_data.get_mosaic_sample_data_path())

    # Which of the two readers applies is decided from the metadata XML, which
    # is parsed here rather than by czifile - worth exercising in Pyodide even
    # though only the mosaic answer can be checked without a multi-view file.
    check(
        "czi_kind_detected",
        czi_utils.is_multiview_czi(czi_path) is False,
        "the sample mosaic must not be read as a multi-view acquisition",
    )

    czi_sources = browser_czi.czi_sources(czi_path)
    check("czi_tiles_enumerated", len(czi_sources) == 2, czi_sources)

    czi_session = Session()
    czi_described = czi_session.load(czi_sources)
    check(
        "czi_session_loaded",
        czi_described["n_views"] == 2
        and {view["served"] for view in czi_described["views"]} == {"virtual"},
        czi_described["views"][0],
    )

    czi_sim = msi_utils.get_sim_from_msim(czi_session.msims[0])
    check(
        "czi_input_is_lazy",
        not isinstance(si_utils._get_backend_data(czi_sim), np.ndarray),
        type(si_utils._get_backend_data(czi_sim)).__name__,
    )

    # Reading pixels is what actually goes through czifile's subblock decoding.
    czi_route = czi_session.view_route(0)
    kind, czi_zarray = czi_session.serve(czi_route, "0/.zarray")
    check("czi_view_zarray", kind == "json", kind)

    czi_key = "/".join("0" for _ in czi_zarray["chunks"])
    kind, czi_chunk = czi_session.serve(czi_route, f"0/{czi_key}")
    czi_expected = int(np.prod(czi_zarray["chunks"])) * np.dtype(
        czi_zarray["dtype"]
    ).itemsize
    check(
        "czi_view_chunk_served",
        kind == "bytes" and len(czi_chunk) == czi_expected,
        f"kind={kind} len={len(czi_chunk) if czi_chunk else None} "
        f"expected={czi_expected}",
    )
    check(
        "czi_view_chunk_has_signal",
        float(
            np.frombuffer(czi_chunk, dtype=np.dtype(czi_zarray["dtype"])).max()
        )
        > 0,
        "decoded subblock is all zeros",
    )

    # A compute worker opens the file for itself from the source URL alone,
    # which is what the page's per-worker mounts exist to make possible.
    czi_worker = WorkerRuntime()
    czi_served = czi_worker.run_task(
        {
            "kind": "serve",
            "session": json.loads(json.dumps(czi_session.spec().to_dict())),
            "route": czi_route,
            "key": f"0/{czi_key}",
        }
    )
    check(
        "czi_chunk_served_by_compute_worker",
        czi_served["kind"] == "bytes" and czi_served["payload"] == czi_chunk,
        czi_served["kind"],
    )

    czi_registered = czi_session.register(
        RegistrationOptions(new_transform_key="registered")
    )
    czi_shift = np.asarray(czi_registered["params"][1]["data"])
    check(
        "czi_registered",
        np.all(np.isfinite(czi_shift)),
        czi_shift.round(2).tolist(),
    )

    # --- the JSON worker API JavaScript actually calls -------------------
    from multiview_stitcher.browser import worker as worker_module

    worker_module._runtime = WorkerRuntime()
    response = json.loads(
        worker_module.handle_json("load", json.dumps({"sources": urls}))
    )
    check("worker_load", response.get("ok"), response)

    response = json.loads(
        worker_module.handle_json(
            "register", json.dumps({"options": {}, "distribute": False})
        )
    )
    check("worker_register", response.get("ok"), response)

    response = json.loads(
        worker_module.handle_json(
            "neuroglancer_state",
            json.dumps({"transform_key": "registered"}),
        )
    )
    check(
        "worker_neuroglancer_state",
        response.get("ok") and len(response["result"]["layers"]) == 2,
        response,
    )

    return json.dumps(
        {
            "ok": True,
            "runtime": info,
            "checks": RESULTS,
            "python": sys.version.split()[0],
        }
    )
