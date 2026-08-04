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

from multiview_stitcher.browser import executors, serialization
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

    def __init__(self, fetch=None, bridge=None):
        self.fetch = fetch
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
        self.session = Session(
            session_id=payload.get("session_id"), fetch=self.fetch
        )
        return self.session.load(payload["sources"])

    def _cmd_describe(self, payload):
        return self._require_session().describe()

    def _cmd_spec(self, payload):
        return self._require_session().spec().to_dict()

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
                    reg_channel_index=options.reg_channel_index,
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
        session.fuse_blocks(plan["options"], plan["block_ids"])
        result = session.finalize_fusion(
            plan["options"], plan["output_stack_properties"]
        )
        result["n_blocks"] = len(plan["block_ids"])
        return result

    def _cmd_transform_keys(self, payload):
        return {"transform_keys": self._require_session().transform_keys()}

    def _cmd_neuroglancer_state(self, payload):
        session = self._require_session()
        return session.neuroglancer_state(
            transform_key=payload.get("transform_key"),
            base_url=payload.get("base_url", ""),
            include_views=payload.get("include_views", True),
            preview_route=payload.get("preview_route"),
            channel_coord=payload.get("channel_coord"),
            contrast_limits=payload.get("contrast_limits"),
            layout=payload.get("layout"),
        )

    # ------------------------------------------------------------------
    # Compute-worker tasks
    # ------------------------------------------------------------------

    def session_for(self, spec):
        """Return a cached read-only session rebuilt from ``spec``."""
        spec = SessionSpec.from_dict(spec)
        key = (
            tuple(source.url for source in spec.sources),
            spec.generation,
        )

        if key not in self._session_cache:
            if len(self._session_cache) >= self.cache_size:
                # Drop the oldest entry; generations only move forward, so the
                # evicted one is the least likely to be asked for again.
                self._session_cache.pop(next(iter(self._session_cache)))
            self._session_cache[key] = Session.from_spec(
                spec, fetch=self.fetch
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
                reg_channel_index=task.get("reg_channel_index"),
            )
        }

    def _task_fuse_blocks(self, task):
        session = self.session_for(task["session"])
        n_blocks = session.fuse_blocks(task["options"], task["block_ids"])
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

    Returns ``(status, content_type, body)`` where ``body`` is ``bytes`` for
    chunks, JSON-encoded ``bytes`` for metadata, and ``None`` for 404s.
    """
    runtime = get_runtime()

    if session_spec is None:
        kind, payload = runtime.serve(route, key)
    else:
        session = runtime.session_for(session_spec)
        kind, payload = session.serve(route, key)

    if kind == "json":
        return (
            200,
            "application/json",
            json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        )
    if kind == "bytes":
        return 200, "application/octet-stream", payload
    return 404, "text/plain", None
