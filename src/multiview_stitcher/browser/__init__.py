"""
Browser execution environment for multiview-stitcher.

This subpackage is *not* a re-implementation: it is a second execution
environment for the same library. Registration, fusion, transformations and
OME-Zarr handling all run through the ordinary
:mod:`multiview_stitcher` functions; what lives here is the thin layer that

* describes work as JSON (:mod:`.specs`, :mod:`.serialization`),
* reads OME-Zarr through the page's service worker (:mod:`.store`,
  :mod:`.dataset`),
* keeps the stateful dataset in one persistent Pyodide worker
  (:mod:`.session`),
* spreads registration and fusion over a pool of workers
  (:mod:`.bridge`, :mod:`.executors`, :mod:`.fusion`),
* supplies the one backend the browser cannot borrow as it is - elastix, as
  WebAssembly rather than as a native extension (:mod:`.elastix`) - and
* exposes a single command/task entry point for JavaScript
  (:mod:`.worker`).

Everything here also runs on CPython, which is what the test suite exercises.
"""

from multiview_stitcher.browser.bridge import (
    Bridge,
    BridgeError,
    LocalBridge,
    TaskError,
    XHRBridge,
    get_bridge,
    set_bridge,
)
from multiview_stitcher.browser.dataset import open_msim, open_msims
from multiview_stitcher.browser.env import is_pyodide, is_worker, runtime_info
from multiview_stitcher.browser.executors import (
    RemoteFusionExecutor,
    RemotePairwiseExecutor,
)
from multiview_stitcher.browser.session import Session
from multiview_stitcher.browser.specs import (
    FusionOptions,
    RegistrationOptions,
    SessionSpec,
    SourceSpec,
)
from multiview_stitcher.browser.store import (
    directory_fetch,
    open_http_store,
    resolve_zarr_source,
)
from multiview_stitcher.browser.worker import (
    WorkerRuntime,
    get_runtime,
    handle_json,
    run_task_json,
    serve_route,
)

__all__ = [
    "Bridge",
    "BridgeError",
    "FusionOptions",
    "LocalBridge",
    "RegistrationOptions",
    "RemoteFusionExecutor",
    "RemotePairwiseExecutor",
    "Session",
    "SessionSpec",
    "SourceSpec",
    "TaskError",
    "WorkerRuntime",
    "XHRBridge",
    "directory_fetch",
    "get_bridge",
    "get_runtime",
    "handle_json",
    "is_pyodide",
    "is_worker",
    "open_http_store",
    "open_msim",
    "open_msims",
    "resolve_zarr_source",
    "run_task_json",
    "runtime_info",
    "serve_route",
    "set_bridge",
]
