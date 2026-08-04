"""
Blocking request/response channel from a Pyodide worker to the host page.

`registration.register` and `fusion.fuse` are ordinary synchronous functions.
To spread their work over a pool of web workers, the worker running them has to
*block* until the pool reports back. Web Workers may do that with a synchronous
``XMLHttpRequest``, which the page's service worker intercepts and answers from
the worker pool. This needs no ``SharedArrayBuffer`` and therefore no COOP/COEP
headers, which matters because GitHub Pages cannot set them.

:class:`LocalBridge` is the same interface backed by ordinary Python calls; it
runs the identical code paths on CPython, which is what the test suite uses.
"""

import json

from multiview_stitcher.browser.env import is_pyodide
from multiview_stitcher.browser.store import FetchError

#: Same-origin prefix owned by the multiview-stitcher service worker. The page
#: overrides this when the app is published below a sub-path, since a service
#: worker can only claim URLs inside its own scope.
DEFAULT_BASE_URL = "/__mvs__"


class Bridge:
    """Interface implemented by all bridges."""

    def call(self, endpoint, payload):
        raise NotImplementedError

    def dispatch(self, tasks):
        """Run ``tasks`` on the worker pool and return their results in order.

        Raises :class:`TaskError` if any task failed.
        """
        response = self.call("dispatch", {"tasks": list(tasks)})
        results = response.get("results", [])

        if len(results) != len(tasks):
            raise TaskError(
                f"worker pool returned {len(results)} results for "
                f"{len(tasks)} tasks"
            )

        errors = [
            result["error"]
            for result in results
            if isinstance(result, dict) and result.get("error")
        ]
        if errors:
            raise TaskError(errors[0] if len(errors) == 1 else str(errors))

        return results


class TaskError(RuntimeError):
    """Raised when a task dispatched to the worker pool failed."""


class XHRBridge(Bridge):  # pragma: no cover - requires a browser worker
    """Bridge over synchronous XHR to the service worker."""

    def __init__(self, base_url=DEFAULT_BASE_URL, session_id=None):
        self.base_url = str(base_url).rstrip("/")
        self.session_id = session_id

    def call(self, endpoint, payload):
        import js

        url = f"{self.base_url}/rpc/{endpoint}"
        if self.session_id:
            url += f"?session={self.session_id}"

        request = js.XMLHttpRequest.new()
        request.open("POST", url, False)
        request.setRequestHeader("Content-Type", "application/json")
        request.send(json.dumps(payload))

        if request.status >= 400:
            raise FetchError(
                f"{request.status} from {url}: {request.responseText}"
            )

        return json.loads(request.responseText)


class LocalBridge(Bridge):
    """Bridge that runs tasks in this process.

    ``runner`` is called once per task and returns that task's result payload.
    An optional ``map_func`` (e.g. a thread pool's ``map``) controls
    concurrency; the default runs tasks sequentially.
    """

    def __init__(self, runner, map_func=None):
        self.runner = runner
        self.map_func = map_func or (lambda func, items: [func(i) for i in items])

    def call(self, endpoint, payload):
        if endpoint != "dispatch":
            raise ValueError(f"LocalBridge cannot serve endpoint '{endpoint}'.")

        def run(task):
            try:
                return self.runner(task)
            except Exception as exc:  # noqa: BLE001 - mirrors the worker pool
                return {"error": f"{type(exc).__name__}: {exc}"}

        return {"results": list(self.map_func(run, payload.get("tasks", [])))}


_bridge = None


def set_bridge(bridge):
    """Install the bridge used by executors in this interpreter."""
    global _bridge
    _bridge = bridge
    return _bridge


def get_bridge():
    """Return the installed bridge, creating the browser default if needed."""
    global _bridge
    if _bridge is None and is_pyodide():  # pragma: no cover - browser only
        _bridge = XHRBridge()
    return _bridge
