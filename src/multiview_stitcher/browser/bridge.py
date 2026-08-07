"""
Blocking request/response channel from a Pyodide worker to the host page.

`registration.register` and `fusion.fuse` are ordinary synchronous functions.
To spread their work over a pool of web workers, the worker running them has to
*block* until the pool reports back. Web Workers may do that with a synchronous
``XMLHttpRequest``, which the page's service worker intercepts and answers from
the worker pool. This needs no ``SharedArrayBuffer`` and therefore no COOP/COEP
headers, which matters because GitHub Pages cannot set them.

Work is *submitted* and then *polled* for, rather than being waited out on one
long request. A browser terminates a service worker whose fetch event outlives
its budget - a few minutes in Chrome - and the request it was answering then
fails with a bare ``NetworkError``. That is a wall the work itself will hit
sooner or later: registering one pair over a long timelapse takes as long as it
takes. Every request here therefore returns within seconds, and the job runs on
in the page, where nothing is watching a clock.

:class:`LocalBridge` is the same interface backed by ordinary Python calls; it
runs the identical code paths on CPython, which is what the test suite uses.
"""

import json
import time
import uuid

from multiview_stitcher.browser.env import is_pyodide
from multiview_stitcher.browser.store import FetchError

#: Same-origin prefix owned by the multiview-stitcher service worker. The page
#: overrides this when the app is published below a sub-path, since a service
#: worker can only claim URLs inside its own scope.
DEFAULT_BASE_URL = "/__mvs__"

#: Tasks per request. Requests no longer stay open while their tasks run, so
#: this only bounds how much JSON one of them carries.
DEFAULT_BATCH_SIZE = 128


class Bridge:
    """Interface implemented by all bridges."""

    def call(self, endpoint, payload):
        raise NotImplementedError

    def run_batch(self, payload):
        """Run one batch of tasks to completion and return ``{"results": ...}``."""
        raise NotImplementedError

    def dispatch(self, tasks, session=None, batch_size=None, progress=None):
        """Run ``tasks`` on the worker pool and return their results in order.

        ``session`` is the session spec every task is rebuilt from. It is sent
        once per request rather than copied into each task: it is by far the
        largest part of the payload, and a job can be thousands of tasks.

        ``progress`` names the job and the unit of work being counted. Each
        request carries how much of the job was already finished before it; the
        page counts the tasks of the current one as they complete, so the bar
        advances per task rather than per request.

        Raises :class:`TaskError` if any task failed.
        """
        tasks = list(tasks)
        if not tasks:
            return []

        size = max(1, int(batch_size or DEFAULT_BATCH_SIZE))
        results = []
        units = [int(task.get("units", 1)) for task in tasks]
        done = 0

        for start in range(0, len(tasks), size):
            batch = tasks[start : start + size]
            # Identified by the caller so that a retried request joins the job
            # it already started instead of running it a second time.
            payload = {"job": uuid.uuid4().hex, "tasks": batch}
            if session is not None:
                payload["session"] = session

            if progress:
                payload["progress"] = {
                    **progress,
                    "completed": done,
                    "total": sum(units),
                }

            response = self.run_batch(payload)
            batch_results = response.get("results", [])

            if len(batch_results) != len(batch):
                raise TaskError(
                    f"worker pool returned {len(batch_results)} results for "
                    f"{len(batch)} tasks"
                )

            errors = [
                result["error"]
                for result in batch_results
                if isinstance(result, dict) and result.get("error")
            ]
            if errors:
                raise TaskError(
                    errors[0] if len(errors) == 1 else str(errors)
                )

            results += batch_results
            done += sum(units[start : start + size])

        return results


class TaskError(RuntimeError):
    """Raised when a task dispatched to the worker pool failed."""


class BridgeError(RuntimeError):
    """Raised when the page could not be reached at all."""


class XHRBridge(Bridge):  # pragma: no cover - requires a browser worker
    """Bridge over synchronous XHR to the service worker."""

    #: How long the page may hold a poll open before answering "still running".
    #: Short enough that no request comes near a service worker's event budget,
    #: long enough that a slow job is not polled thousands of times.
    poll_timeout_ms = 4000

    #: A dropped request is retried rather than failing the job: the browser is
    #: free to recycle the service worker between two of them, and both
    #: endpoints are idempotent so a repeat costs nothing.
    max_attempts = 6

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

    def run_batch(self, payload):
        """Hand the batch to the page, then poll until it has finished."""
        self._call_resiliently("dispatch", payload)

        while True:
            response = self._call_resiliently(
                "poll",
                {"job": payload["job"], "timeout_ms": self.poll_timeout_ms},
            )
            if response.get("done"):
                return response

    def _call_resiliently(self, endpoint, payload):
        """Call ``endpoint``, retrying a request that never reached the page.

        Only transport failures are retried. A response *from* the page - any
        4xx or 5xx - is an answer, and repeating the question cannot change it.
        """
        delay = 0.25
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self.call(endpoint, payload)
            except FetchError:
                raise
            except Exception as exc:  # noqa: BLE001 - JsException, typically
                if attempt == self.max_attempts:
                    raise BridgeError(
                        f"the page did not answer '{endpoint}' after "
                        f"{self.max_attempts} attempts: {exc}"
                    ) from exc
                time.sleep(delay)
                delay = min(delay * 2, 4.0)
        raise AssertionError("unreachable")


class LocalBridge(Bridge):
    """Bridge that runs tasks in this process.

    ``runner`` is called once per task and returns that task's result payload.
    An optional ``map_func`` (e.g. a thread pool's ``map``) controls
    concurrency; the default runs tasks sequentially.
    """

    def __init__(self, runner, map_func=None):
        self.runner = runner
        self.map_func = map_func or (lambda func, items: [func(i) for i in items])

    def run_batch(self, payload):
        return self.call("dispatch", payload)

    def call(self, endpoint, payload):
        if endpoint != "dispatch":
            raise ValueError(f"LocalBridge cannot serve endpoint '{endpoint}'.")

        # The session travels once per request; each task is given it here,
        # exactly as the page does before handing a task to a compute worker.
        session = payload.get("session")

        def run(task):
            if session is not None and task.get("session") is None:
                task = {**task, "session": session}
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
