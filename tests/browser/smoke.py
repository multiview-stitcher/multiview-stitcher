"""
End-to-end smoke test of the browser runtime, executed inside Pyodide.

It walks the whole path the browser app takes - write a miniature multiscale
OME-Zarr, open it through the browser session, register two views, fuse them
lazily and read one fused chunk - and checks the results, so that platform
differences between CPython and Pyodide (zarr v2, older xarray, no ngff-zarr
or ome-zarr-py) surface in CI rather than in the UI.

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


def build_dataset(root="/data"):
    """Write two overlapping tiles as multiscale OME-Zarr v0.4."""
    import os

    from multiview_stitcher import msi_utils, ngff_utils, sample_data

    os.makedirs(root, exist_ok=True)

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
        url = f"{root}/tile_{index}.ome.zarr"
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
        "zarr_v2",
        zarr.__version__.startswith("2."),
        f"zarr {zarr.__version__}",
    )
    # The reference NGFF packages have no WebAssembly build; the built-in
    # readers/writers must carry the OME-Zarr handling here.
    check(
        "ngff_zarr_absent",
        ngff_utils.ngff_zarr is None,
        "ngff-zarr unexpectedly importable",
    )
    check(
        "ome_zarr_absent",
        ngff_utils.writer is None,
        "ome-zarr unexpectedly importable",
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
            reg_channel_index=0,
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
    check(
        "stale_route_invalidated",
        session.serve(route, ".zattrs") == ("missing", None),
        "old generation still served data",
    )

    # --- fuse to an OME-Zarr on disk ------------------------------------
    options = FusionOptions(
        transform_key="registered", output_zarr_url="/data/fused.ome.zarr"
    )
    plan = session.fusion_plan(options)
    session.fuse_blocks(plan["options"], plan["block_ids"])
    session.finalize_fusion(plan["options"], plan["output_stack_properties"])

    fused_msim = ngff_utils.read_msim_from_ome_zarr("/data/fused.ome.zarr")
    fused_sim = msi_utils.get_sim_from_msim(fused_msim)
    expected_shape = plan["output_stack_properties"]["shape"]
    check(
        "fused_zarr_shape",
        all(
            fused_sim.sizes[dim] == size
            for dim, size in expected_shape.items()
        ),
        f"{dict(fused_sim.sizes)} vs {expected_shape}",
    )
    check(
        "fused_zarr_has_signal",
        float(np.asarray(fused_sim.data).max()) > 0,
        "fused output is empty",
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
