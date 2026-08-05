"""Runtime detection for the browser execution environment."""

import platform
import sys


def is_pyodide():
    """True when running inside Pyodide (CPython compiled to WebAssembly)."""
    return sys.platform == "emscripten"


def is_worker():
    """True when the Pyodide runtime lives in a Web Worker rather than the page.

    Only Web Workers may issue synchronous XHR, which is how the Python side
    performs blocking reads through the service worker.
    """
    if not is_pyodide():
        return False
    try:
        import js
    except ImportError:  # pragma: no cover - only reachable outside Pyodide
        return False
    # DedicatedWorkerGlobalScope has importScripts(); the window scope has not.
    return hasattr(js, "importScripts")


def runtime_info():
    """Small dict describing the runtime, surfaced in the browser UI."""
    import numpy as np

    info = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "machine": platform.machine(),
        "pyodide": is_pyodide(),
        "worker": is_worker(),
        "numpy": np.__version__,
    }

    for name in ("zarr", "dask", "xarray", "scipy", "skimage", "networkx"):
        try:
            info[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001 - report, never fail
            info[name] = None

    try:
        from multiview_stitcher import __version__

        info["multiview_stitcher"] = __version__
    except ImportError:  # pragma: no cover
        info["multiview_stitcher"] = None

    info["zarr_sync"] = check_zarr_sync()

    return info


def check_zarr_sync():
    """Report whether zarr's synchronous API can run in this runtime.

    zarr v3 is asynchronous underneath. On CPython it blocks by running an
    event loop on another thread; in the browser there is no thread, so it
    blocks by suspending the WebAssembly stack instead - which only Pyodide's
    own build of zarr knows how to do. The wheel of the same version on PyPI
    does not, and every read fails with "can't start new thread" at whatever
    moment the user first opens an image.

    Checked at start-up so the answer arrives with the version banner rather
    than as a traceback halfway through a session. Returns "ok", or a string
    naming the problem.
    """
    import zarr

    try:
        zarr.create_array(
            store={}, shape=(1,), chunks=(1,), dtype="uint8", overwrite=True
        )
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        message = str(exc)
        if "start new thread" in message:
            hint = (
                "this build of zarr blocks by starting a thread, and the "
                "browser has none. zarr must come from Pyodide's own package "
                "index rather than PyPI: only that build suspends the "
                "WebAssembly stack instead."
            )
        elif "stack switch" in message:
            hint = (
                "zarr can block here, but only when Python was entered in a "
                "way that may suspend. Call it with `callPromising` - see "
                "docs/browser/py-runtime.js."
            )
        else:
            hint = "zarr's synchronous API does not work in this runtime."
        return f"{type(exc).__name__}: {exc}. {hint}"

    return "ok"
