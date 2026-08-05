"""
The single entry point JavaScript calls into.

Two roles share this module:

* the **session worker** owns the authoritative :class:`Session` and answers
  UI commands (`handle`);
* every **compute worker** answers stateless tasks (`run_task`), rebuilding a
  read-only session from the spec carried in the task and caching it per
  session generation so repeated tasks stay cheap.

Both roles serve virtual OME-Zarr chunk requests, which is what makes the
lazily fused preview render in parallel.
"""

import json
import traceback

from multiview_stitcher.browser import example_data, executors, serialization
from multiview_stitcher.browser.bridge import get_bridge
from multiview_stitcher.browser.env import runtime_info
from multiview_stitcher.browser.session import Session
from multiview_stitcher.browser.specs import (
    FusionOptions,
    RegistrationOptions,
    SessionSpec,
)


class WorkerRuntime:
    """Command and task dispatch for one Pyodide worker."""

    #: How many rebuilt sessions a compute worker keeps around.
    cache_size = 2

    def __init__(self, fetch=None, write=None, bridge=None):
        self.fetch = fetch
        self.write = write
        self.bridge = bridge
        self.session = None
        self._session_cache = {}

    # ------------------------------------------------------------------
    # Session-worker commands
    # ------------------------------------------------------------------

    def handle(self, command, payload=None):
        payload = payload or {}
        handler = getattr(self, f"_cmd_{command}", None)
        if handler is None:
            raise ValueError(f"Unknown command '{command}'.")
        return handler(payload)

    def _cmd_info(self, payload):
        return runtime_info()

    def _require_session(self):
        if self.session is None:
            raise RuntimeError("No dataset has been loaded yet.")
        return self.session

    def _cmd_load(self, payload):
        """Open sources, replacing or extending whatever is already loaded."""
        replace = payload.get("replace", True)

        if self.session is None or replace:
            # Swap only once the new session has opened successfully. Failing
            # halfway used to leave an empty session in place of a working
            # one, after which every URL the viewer still held answered "not
            # found" - an image that silently emptied itself because an
            # unrelated load failed.
            session = Session(
                session_id=payload.get("session_id"),
                fetch=self.fetch,
                write=self.write,
            )
            described = session.load(payload["sources"])
            self.session = session
            return described

        return self.session.add(payload["sources"])

    def _cmd_load_example(self, payload):
        """Load one of the generated example datasets."""
        name = payload.get("name", "tiles-3d")
        if name not in example_data.EXAMPLES:
            raise ValueError(
                f"Unknown example '{name}'. Available: "
                f"{sorted(example_data.EXAMPLES)}."
            )
        return self._cmd_load(
            {
                "sources": example_data.example_sources(name),
                "replace": payload.get("replace", True),
            }
        )

    def _cmd_examples(self, payload):
        return {
            "examples": [
                {"name": name, "label": example_data.EXAMPLES[name]["label"]}
                for name in example_data.EXAMPLE_MENU
            ]
        }

    def _cmd_remove(self, payload):
        return self._require_session().remove(payload["index"])

    def _cmd_clear(self, payload):
        return self._require_session().clear()

    def _cmd_describe(self, payload):
        return self._require_session().describe()

    def _cmd_spec(self, payload):
        return self._require_session().spec().to_dict()

    def _cmd_copy_transform(self, payload):
        return self._require_session().copy_transform(
            payload.get("source_transform_key"),
            payload.get("new_transform_key"),
        )

    def _cmd_update_transforms(self, payload):
        return self._require_session().update_neuroglancer_transforms(
            payload.get("transform_key"), payload.get("updates", [])
        )

    def _cmd_register(self, payload):
        session = self._require_session()
        options = RegistrationOptions.from_dict(payload.get("options"))

        pairwise_executor = None
        if payload.get("distribute", True):
            bridge = self.bridge or get_bridge()
            if bridge is not None:
                pairwise_executor = executors.RemotePairwiseExecutor(
                    session.spec(),
                    bridge=bridge,
                    max_pairs_per_task=int(
                        payload.get("pairs_per_task", 1) or 1
                    ),
                )

        return session.register(
            options, pairwise_executor=pairwise_executor
        )

    def _cmd_fuse_preview(self, payload):
        return self._require_session().fuse_preview(payload.get("options"))

    def _cmd_fuse_to_zarr(self, payload):
        session = self._require_session()
        options = FusionOptions.from_dict(payload.get("options"))
        if options.output_zarr_url is None:
            raise ValueError(
                "Fusing to disk needs an output_zarr_url; use fuse_preview "
                "for the lazy in-viewer fusion."
            )

        plan = session.fusion_plan(options)

        # Every block of every level is an independent chunk file, so the pool
        # can write them all at once into the one output directory.
        executor = None
        if payload.get("distribute", True):
            bridge = self.bridge or get_bridge()
            if bridge is not None:
                executor = executors.RemoteFusionExecutor(
                    session.spec(),
                    bridge=bridge,
                    n_workers=int(payload.get("n_workers", 1) or 1),
                )

        if executor is not None:
            n_blocks = executor(plan["options"], plan["levels"])
        else:
            n_blocks = sum(
                session.fuse_blocks(
                    plan["options"], level["level"], level["block_ids"]
                )
                for level in plan["levels"]
            )

        result = session.finalize_fusion(plan["options"])
        result["n_blocks"] = n_blocks
        return result

    def _cmd_transform_keys(self, payload):
        return {"transform_keys": self._require_session().transform_keys()}

    def _cmd_neuroglancer_state(self, payload):
        session = self._require_session()
        return session.neuroglancer_state(
            transform_key=payload.get("transform_key"),
            base_url=payload.get("base_url", ""),
            api_base=payload.get("api_base", ""),
            serve_views=payload.get("serve_views", "auto"),
            include_views=payload.get("include_views", True),
            preview_route=payload.get("preview_route"),
            channel_coord=payload.get("channel_coord"),
            contrast_limits=payload.get("contrast_limits"),
            layout=payload.get("layout"),
            show_all_channels=payload.get("show_all_channels", False),
        )

    # ------------------------------------------------------------------
    # Compute-worker tasks
    # ------------------------------------------------------------------

    def session_for(self, spec):
        """Return a cached read-only session rebuilt from ``spec``."""
        spec = SessionSpec.from_dict(spec)
        # The preview belongs in the key: a cached session rebuilt before a
        # preview existed cannot serve it, and answering "not found" would
        # render as an empty layer rather than as an error.
        key = (
            tuple(source.url for source in spec.sources),
            spec.generation,
            json.dumps(spec.preview, sort_keys=True),
            # A registration adds transforms without changing what any view
            # route serves, so it deliberately does not move the generation.
            # They still have to be part of the key: fusing reads them, and a
            # session cached before a registration would otherwise go on
            # fusing with the transforms it was built with.
            json.dumps(spec.transforms, sort_keys=True),
        )

        if key not in self._session_cache:
            if len(self._session_cache) >= self.cache_size:
                # Drop the oldest entry; generations only move forward, so the
                # evicted one is the least likely to be asked for again.
                self._session_cache.pop(next(iter(self._session_cache)))
            self._session_cache[key] = Session.from_spec(
                spec, fetch=self.fetch, write=self.write
            )

        return self._session_cache[key]

    def invalidate(self):
        """Drop every cached session (used when inputs change)."""
        self._session_cache.clear()

    def run_task(self, task):
        kind = task.get("kind")
        runner = getattr(self, f"_task_{kind}", None)
        if runner is None:
            raise ValueError(f"Unknown task kind '{kind}'.")
        return runner(task)

    def _task_register_pairs(self, task):
        session = self.session_for(task["session"])
        register_kwargs = executors.deserialize_register_kwargs(
            task["register_kwargs"]
        )
        return {
            "pairwise": session.compute_pairwise(
                task["edges"],
                register_kwargs,
                reg_channel=task.get("reg_channel"),
            )
        }

    def _task_fuse_blocks(self, task):
        session = self.session_for(task["session"])
        n_blocks = session.fuse_blocks(
            task["options"], task["level"], task["block_ids"]
        )
        return {"n_blocks": n_blocks}

    def _task_serve(self, task):
        """Serve a virtual OME-Zarr request for a session this worker rebuilt."""
        session = self.session_for(task["session"])
        kind, payload = session.serve(task["route"], task["key"])
        return {"kind": kind, "payload": payload}

    # ------------------------------------------------------------------
    # Serving from the session worker
    # ------------------------------------------------------------------

    def serve(self, route, key):
        if self.session is None:
            return "missing", None
        return self.session.serve(route, key)


_runtime = None


def get_runtime(**kwargs):
    """Return this worker's runtime, creating it on first use."""
    global _runtime
    if _runtime is None:
        _runtime = WorkerRuntime(**kwargs)
    return _runtime


def _error_payload(exc):
    return {
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
    }


def handle_json(command, payload_json="{}"):
    """JSON-in / JSON-out command dispatch, called from JavaScript.

    Errors are returned rather than raised so the JavaScript side always gets a
    structured response it can show in the UI.
    """
    try:
        payload = json.loads(payload_json) if payload_json else {}
        result = get_runtime().handle(command, payload)
        return json.dumps(
            {"ok": True, "result": serialization.to_jsonable(result)}
        )
    except Exception as exc:  # noqa: BLE001 - reported to the UI
        return json.dumps({"ok": False, **_error_payload(exc)})


def run_task_json(task_json):
    """JSON-in / JSON-out task dispatch, called from JavaScript."""
    try:
        task = json.loads(task_json)
        result = get_runtime().run_task(task)
        return json.dumps(
            {"ok": True, "result": serialization.to_jsonable(result)}
        )
    except Exception as exc:  # noqa: BLE001 - reported to the UI
        return json.dumps({"ok": False, **_error_payload(exc)})


def serve_route(route, key, session_spec=None):
    """Answer one virtual OME-Zarr request.

    ``session_spec`` is a JSON string, matching the other entry points. Taking
    a live JavaScript object instead would let its nulls arrive as `JsNull`
    proxies rather than None, which satisfy an ``is not None`` check and then
    fail deep inside numeric code.

    Returns ``(status, content_type, body)`` where ``body`` is ``bytes`` for
    chunks, JSON-encoded ``bytes`` for metadata, and the reason for 404s.
    """
    runtime = get_runtime()

    if isinstance(session_spec, str):
        session_spec = json.loads(session_spec) if session_spec else None

    try:
        session = None
        if session_spec is not None:
            try:
                session = runtime.session_for(session_spec)
            except ValueError:
                # A page can hand over a spec this worker cannot rebuild from,
                # for instance when a cached script and a fresh wheel disagree.
                # Answering from this worker's own session when it has one -
                # and failing loudly when it does not, so the page retries
                # elsewhere - beats refusing outright.
                if runtime.session is None:
                    raise
                session = runtime.session

        if session is None:
            kind, payload = runtime.serve(route, key)
        else:
            kind, payload = session.serve(route, key)
    except Exception as exc:  # noqa: BLE001 - reported over HTTP
        # Reported as a server error rather than "not found": zarr reads a
        # missing chunk as its fill value, so a failure answered with 404
        # renders as a black image and is never seen.
        return (
            500,
            "text/plain",
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}".encode(),
        )

    if kind == "json":
        return (
            200,
            "application/json",
            json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        )
    if kind == "bytes":
        return 200, "application/octet-stream", payload

    return 404, "text/plain", str(payload or "not found").encode("utf-8")
