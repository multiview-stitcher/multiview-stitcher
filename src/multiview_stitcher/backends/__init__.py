"""Backend system for multiview-stitcher.

Backends
--------
Array API backends (unified code path):
    ``"numpy"``  -- default, CPU via numpy + scipy
    ``"cupy"``   -- NVIDIA GPU via cupy + cupyx  (auto-registered when importable)
    ``"dpnp"``   -- Intel XPU via dpnp           (auto-registered when importable)

Legacy backends (original implementations, kept for A/B benchmarking):
    ``"numpy-legacy"`` -- original NumpyLegacyBackend
    ``"cupy-legacy"``  -- original CupyLegacyBackend  (auto-registered when importable)

Platform-specific backends:
    ``"jax"``    -- CPU / CUDA GPU / TPU via JAX (auto-registered when importable)
    ``"mlx"``    -- Apple Silicon GPU via MLX    (auto-registered on macOS)
"""

from multiview_stitcher.backends._array_api import ArrayAPIBackend
from multiview_stitcher.backends._base import Backend
from multiview_stitcher.backends._numpy_legacy import NumpyLegacyBackend

_GLOBAL_BACKEND: str = "numpy"

_REGISTRY: dict[str, object] = {
    "numpy": lambda: ArrayAPIBackend("numpy"),
    "numpy-legacy": NumpyLegacyBackend,
}


def get_backend(name: str | None = None) -> Backend:
    """Return an instantiated backend by name.

    If name is None, return the global default.
    """
    if name is None:
        name = _GLOBAL_BACKEND
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown backend {name!r}. Available: {list(_REGISTRY)}"
        )
    entry = _REGISTRY[name]
    if isinstance(entry, type):
        return entry()
    return entry()


def set_backend(name: str) -> None:
    """Set the global default backend."""
    global _GLOBAL_BACKEND
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown backend {name!r}. Available: {list(_REGISTRY)}"
        )
    _GLOBAL_BACKEND = name


def register_backend(name: str, backend_cls: type) -> None:
    """Register a new backend class."""
    _REGISTRY[name] = backend_cls


# Auto-register CuPy backends if available
try:
    from multiview_stitcher.backends._cupy_legacy import CupyLegacyBackend

    _REGISTRY["cupy"] = lambda: ArrayAPIBackend("cupy")
    _REGISTRY["cupy-legacy"] = CupyLegacyBackend
except Exception:
    pass

# Auto-register dpnp backend if available
try:
    import dpnp as _dpnp  # noqa: F401

    _REGISTRY["dpnp"] = lambda: ArrayAPIBackend("dpnp")
    del _dpnp
except Exception:
    pass

# Auto-register JAX backend if available
try:
    import jax as _jax  # noqa: F401

    from multiview_stitcher.backends._jax_backend import JaxBackend

    _REGISTRY["jax"] = JaxBackend
    del _jax
except Exception:
    pass

# Auto-register MLX backend if available (macOS Apple Silicon)
try:
    import mlx.core as _mlx_core  # noqa: F401

    from multiview_stitcher.backends._mlx_backend import MLXBackend

    _REGISTRY["mlx"] = MLXBackend
    del _mlx_core
except Exception:
    pass

def __getattr__(name):
    """Lazy-load benchmark utilities to avoid importing the entire library at startup."""
    _benchmark_names = {
        "evaluate_benchmarks",
        "max_tile_size",
        "plan_benchmarks",
        "plot_runtime_and_speedup",
        "run_benchmarks",
    }
    if name in _benchmark_names:
        from multiview_stitcher.backends._benchmark import (
            evaluate_benchmarks,
            max_tile_size,
            plan_benchmarks,
            plot_runtime_and_speedup,
            run_benchmarks,
        )
        _lazy = {
            "evaluate_benchmarks": evaluate_benchmarks,
            "max_tile_size": max_tile_size,
            "plan_benchmarks": plan_benchmarks,
            "plot_runtime_and_speedup": plot_runtime_and_speedup,
            "run_benchmarks": run_benchmarks,
        }
        return _lazy[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
