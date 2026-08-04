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

    return info
