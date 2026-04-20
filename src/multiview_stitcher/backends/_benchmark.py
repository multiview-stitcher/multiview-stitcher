"""Benchmark utilities for the backend system.

Three-phase API::

    from multiview_stitcher.backends import plan_benchmarks, run_benchmarks

    # Phase 1: plan what to benchmark
    plan = plan_benchmarks(
        specs=["numpy", "cupy:gpu:0"],
        tile_sizes=[100, 200],
        functions=["transform_data", "fuse"],
    )

    # Phase 2: run and write results incrementally
    result = run_benchmarks(plan, output_path="bench_results/", n_runs=3)

    # Phase 3: load, merge, and plot
    from multiview_stitcher.backends import evaluate_benchmarks, plot_runtime_and_speedup
    merged = evaluate_benchmarks("bench_results/")
    plot_runtime_and_speedup(merged)

Backend spec syntax::

    "backend_name[:device_type[:device_id]]"

    Examples: "numpy", "cupy:gpu:0", "jax:cpu", "dpnp:xpu:0"
"""

import json
import os
import platform
import tempfile
import time
from datetime import datetime, timezone

import numpy as np

import dask

import multiview_stitcher
from multiview_stitcher import (
    fusion,
    msi_utils,
    registration,
    sample_data,
    transformation,
    weights,
)
from multiview_stitcher import param_utils
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher.backends import _REGISTRY, get_backend
from multiview_stitcher.io import METADATA_TRANSFORM_KEY


# ── System info ──────────────────────────────────────────────────────


def _gather_system_info():
    """Collect system information for traceability."""
    import hashlib

    info = {
        "id": None,  # set below from hardware fingerprint
        "platform": platform.platform(),
        "python": platform.python_version(),
        "multiview_stitcher": multiview_stitcher.__version__,
        "numpy": np.__version__,
        "cpu": "unknown",
        "ram_gb": None,
        "gpus": [],
    }

    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except FileNotFoundError:
        info["cpu"] = platform.processor() or platform.machine()

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    info["ram_gb"] = round(int(line.split()[1]) / 1e6, 1)
                    break
    except FileNotFoundError:
        pass

    if "cupy" in _REGISTRY:
        try:
            import cupy as _cp

            info["cupy"] = _cp.__version__
            for dev_id in range(_cp.cuda.runtime.getDeviceCount()):
                props = _cp.cuda.runtime.getDeviceProperties(dev_id)
                mem = _cp.cuda.Device(dev_id).mem_info
                info["gpus"].append(
                    {
                        "id": dev_id,
                        "name": props["name"].decode(),
                        "memory_total_gb": round(mem[1] / 1e9, 1),
                        "memory_free_gb": round(mem[0] / 1e9, 1),
                    }
                )
        except Exception:
            pass

    if "dpnp" in _REGISTRY:
        try:
            import dpnp as _dpnp
            import dpctl

            info["dpnp"] = _dpnp.__version__
            for dev in dpctl.get_devices():
                if dev.backend.name != "host":
                    info["gpus"].append(
                        {
                            "id": 0,
                            "name": dev.name,
                            "backend": dev.backend.name,
                        }
                    )
        except Exception:
            pass

    for pkg in ("numba", "jax"):
        if pkg in _REGISTRY:
            try:
                mod = __import__(pkg)
                info[pkg] = mod.__version__
            except Exception:
                pass

    if "jax" in _REGISTRY and not info["gpus"]:
        try:
            import jax as _jax
            for dev in _jax.devices():
                if dev.platform != "cpu":
                    info["gpus"].append({
                        "id": dev.id,
                        "name": dev.device_kind,
                    })
        except Exception:
            pass

    _fp_parts = [info["cpu"], str(info.get("ram_gb", ""))]
    info["id"] = hashlib.sha256(
        "|".join(_fp_parts).encode()
    ).hexdigest()[:12]

    return info


# ── Backend classification helpers ───────────────────────────────────

_CUPY_BACKEND_NAMES = {"cupy", "cupy-legacy"}
_CPU_BACKEND_NAMES = {"numpy", "numpy-legacy"}


def _is_cupy_backend(name):
    return name in _CUPY_BACKEND_NAMES


def _is_gpu_backend(name):
    if name in _CUPY_BACKEND_NAMES or name == "dpnp":
        return True
    if name in ("jax", "mlx"):
        return True
    return False


def _free_gpu_pool():
    """Release cached CuPy memory so the next function starts fresh."""
    try:
        import cupy as _cp
        _cp.cuda.Device().synchronize()
        _cp.get_default_memory_pool().free_all_blocks()
        _cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def _get_memory_usage_mb(backend_name):
    """Return (used_MB, total_MB) for the appropriate device.

    For GPU backends the GPU memory is reported; for CPU backends the
    process RSS is returned (with total system RAM as the second value).
    """
    if _is_gpu_backend(backend_name):
        # --- CuPy / cupy-legacy ---
        if _is_cupy_backend(backend_name):
            try:
                import cupy as _cp
                free, total = _cp.cuda.Device().mem_info
                used = total - free
                return used / 1024**2, total / 1024**2
            except Exception:
                pass

        # --- JAX ---
        if backend_name == "jax":
            try:
                import jax
                dev = jax.devices()[0]
                stats = dev.memory_stats()
                if stats is not None:
                    used = stats.get("bytes_in_use", 0)
                    total = stats.get("bytes_limit", 0)
                    return used / 1024**2, total / 1024**2
            except Exception:
                pass

        # --- dpnp / Intel GPU ---
        if backend_name == "dpnp":
            try:
                import dpctl
                dev = dpctl.SyclDevice()
                total = dev.global_mem_size
                free = dev.global_mem_free  # may not exist on all runtimes
                return (total - free) / 1024**2, total / 1024**2
            except Exception:
                pass

        # --- MLX (Apple) ---
        if backend_name == "mlx":
            try:
                import mlx.core as mx
                used = mx.metal.get_active_memory()
                total = mx.metal.get_cache_memory()
                return used / 1024**2, total / 1024**2
            except Exception:
                pass

    # Fallback: CPU process RSS + system RAM
    try:
        import resource
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On Linux ru_maxrss is in KB; on macOS it's bytes
        if platform.system() == "Darwin":
            rss_mb = rss_kb / 1024**2
        else:
            rss_mb = rss_kb / 1024
        total_mb = None
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        total_mb = int(line.split()[1]) / 1024
                        break
        except FileNotFoundError:
            pass
        return rss_mb, total_mb
    except Exception:
        return None, None


def _fmt_mem(used_mb, total_mb):
    """Format a (used, total) memory pair as a compact string."""
    if used_mb is None:
        return "N/A"
    used_str = f"{used_mb:.0f}MB"
    if total_mb is not None:
        return f"{used_str}/{total_mb:.0f}MB"
    return used_str


# ── Backend spec parsing ─────────────────────────────────────────────

_VALID_DEVICE_TYPES = {"cpu", "gpu", "xpu", "tpu"}


def parse_backend_spec(spec):
    """Parse a backend spec string into components.

    Format: ``"backend_name[:device_type[:device_id]]"``

    Parameters
    ----------
    spec : str
        Backend specification string.

    Returns
    -------
    dict
        Keys: ``backend_name`` (str), ``device_type`` (str or None),
        ``device_id`` (int or None).

    Raises
    ------
    ValueError
        If the spec is malformed or the backend is not registered.
    """
    parts = spec.split(":")
    if len(parts) > 3:
        raise ValueError(
            f"Invalid backend spec {spec!r}: expected at most 3 colon-separated "
            f"fields (backend_name:device_type:device_id)"
        )

    backend_name = parts[0]
    if backend_name not in _REGISTRY:
        raise ValueError(
            f"Unknown backend {backend_name!r} in spec {spec!r}. "
            f"Available: {list(_REGISTRY)}"
        )

    device_type = None
    device_id = None

    if len(parts) >= 2 and parts[1]:
        device_type = parts[1].lower()
        if device_type not in _VALID_DEVICE_TYPES:
            raise ValueError(
                f"Invalid device_type {parts[1]!r} in spec {spec!r}. "
                f"Expected one of: {sorted(_VALID_DEVICE_TYPES)}"
            )

    if len(parts) >= 3 and parts[2]:
        try:
            device_id = int(parts[2])
        except ValueError:
            raise ValueError(
                f"Invalid device_id {parts[2]!r} in spec {spec!r}. "
                f"Expected an integer."
            )
        if device_id < 0:
            raise ValueError(
                f"device_id must be non-negative, got {device_id} "
                f"in spec {spec!r}"
            )

    return {
        "backend_name": backend_name,
        "device_type": device_type,
        "device_id": device_id,
    }


# ── Device enumeration ───────────────────────────────────────────────


def _enumerate_cupy_devices(device_type=None, device_id=None):
    """Return ``[(device_id, device_label), ...]`` for CuPy backends."""
    import cupy as _cp

    n_devices = _cp.cuda.runtime.getDeviceCount()
    all_devices = [(i, f"gpu:{i}") for i in range(n_devices)]

    # CuPy devices are always GPUs
    if device_type is not None and device_type != "gpu":
        return []

    if device_id is not None:
        return [(d, l) for d, l in all_devices if d == device_id]
    return all_devices


def _enumerate_dpnp_devices(device_type=None, device_id=None):
    """Return ``[(device_id, device_label), ...]`` for dpnp backends."""
    try:
        import dpctl
    except ImportError:
        return [(0, "xpu:0")]

    all_devs = [d for d in dpctl.get_devices() if d.backend.name != "host"]

    # Filter by device_type
    if device_type is not None:
        type_map = {"gpu": "gpu", "cpu": "cpu", "xpu": "gpu"}
        target = type_map.get(device_type, device_type)
        all_devs = [d for d in all_devs if d.device_type.name.lower() == target]

    if not all_devs:
        return []

    devices = [(i, f"xpu:{i}") for i in range(len(all_devs))]

    if device_id is not None:
        return [(d, l) for d, l in devices if d == device_id]
    return devices


def _enumerate_jax_devices(device_type=None, device_id=None):
    """Return ``[(device_id, device_label), ...]`` for JAX backends."""
    try:
        import jax as _jax
        jax_devices = _jax.devices()
    except Exception:
        return [(0, "cpu:0")]

    result = []
    for dev in jax_devices:
        if device_type is not None:
            if device_type == "cpu" and dev.platform != "cpu":
                continue
            if device_type == "gpu" and dev.platform not in ("cuda", "rocm", "gpu"):
                continue
            if device_type == "tpu" and dev.platform != "tpu":
                continue

        if dev.platform == "cpu":
            label = "cpu:0"
        else:
            label = f"gpu:{dev.id}"

        result.append((dev.id, label))

    if device_id is not None:
        result = [(d, l) for d, l in result if d == device_id]

    return result if result else [(0, "cpu:0")]


def _enumerate_devices(backend_name, device_type=None, device_id=None):
    """Enumerate available devices for a backend.

    Returns
    -------
    list of (int, str)
        List of ``(device_id, device_label)`` tuples.
    """
    if _is_cupy_backend(backend_name):
        return _enumerate_cupy_devices(device_type, device_id)
    if backend_name == "dpnp":
        return _enumerate_dpnp_devices(device_type, device_id)
    if backend_name == "jax":
        return _enumerate_jax_devices(device_type, device_id)
    if backend_name == "mlx":
        return [(0, "cpu:0")]
    # CPU backends: numpy, numpy-legacy
    return [(0, "cpu:0")]


# ── Input preparation ────────────────────────────────────────────────

# Estimated number of concurrent float32 arrays per function.
_CONCURRENT_ARRAYS = {
    "transform_data": 3,
    "get_blending_weights": 8,
    "register": 4,
    "fuse": 16,
}
_CONCURRENT_ARRAYS_ROCM = {
    "transform_data": 4,
    "get_blending_weights": 12,
    "register": 5,
    "fuse": 24,
}
# XLA keeps coordinate grids, interpolation weights, gathered neighbours,
# and various intermediate buffers alive simultaneously.  The peak memory
# multiplier differs by platform because XLA's compilation passes, fusion
# heuristics, and downstream BLAS paths (rocBLAS vs Triton GEMM vs TPU
# systolic) all produce different intermediate buffer patterns.
#
# ROCm values empirically measured on AMD Instinct MI100 (gfx908) under
# ROCm 7.2 via jax.devices()[0].memory_stats()["peak_bytes_in_use"] across
# chunk sizes 64–400.  Peak converges to ~35× single-array size for
# transform_data / get_blending_weights / fuse, and ~10× for register.
_CONCURRENT_ARRAYS_JAX_ROCM = {
    "transform_data": 40,
    "get_blending_weights": 40,
    "register": 12,
    "fuse": 40,
}
# CUDA / TPU / CPU values retain the initial rough estimates — they have
# not yet been empirically re-measured on those platforms.  Replace with
# measured values when hardware is available.
_CONCURRENT_ARRAYS_JAX_CUDA = {
    "transform_data": 20,
    "get_blending_weights": 20,
    "register": 10,
    "fuse": 32,
}
_CONCURRENT_ARRAYS_JAX_TPU = dict(_CONCURRENT_ARRAYS_JAX_CUDA)
_CONCURRENT_ARRAYS_JAX_CPU = dict(_CONCURRENT_ARRAYS_JAX_CUDA)

_CONCURRENT_ARRAYS_JAX_BY_PLATFORM = {
    "rocm": _CONCURRENT_ARRAYS_JAX_ROCM,
    "cuda": _CONCURRENT_ARRAYS_JAX_CUDA,
    "tpu": _CONCURRENT_ARRAYS_JAX_TPU,
    "cpu": _CONCURRENT_ARRAYS_JAX_CPU,
}


def _detect_jax_platform():
    """Return 'rocm', 'cuda', 'tpu', or 'cpu' for the active JAX backend."""
    try:
        import jax
        devs = jax.devices()
        if not devs:
            return "cpu"
        platform = devs[0].platform  # 'cpu', 'gpu', 'tpu'
        if platform in ("cpu", "tpu"):
            return platform
        # GPU — distinguish CUDA vs ROCm via the XLA backend's version string.
        try:
            backend = jax.extend.backend.get_backend()
            ver = getattr(backend, "platform_version", "").lower()
            if "rocm" in ver or "hip" in ver:
                return "rocm"
            if "cuda" in ver:
                return "cuda"
        except Exception:
            pass
        # Fallback: infer from device marketing name.
        kind = str(devs[0].device_kind).lower()
        if any(k in kind for k in ("amd", "radeon", "instinct")):
            return "rocm"
        return "cuda"
    except Exception:
        return "cpu"
_PEAK_FLOAT32_ARRAYS = 16


def max_tile_size(
    ndim=3,
    dtype_bytes=4,
    device_id=0,
    safety_factor=0.85,
    n_concurrent_arrays=_PEAK_FLOAT32_ARRAYS,
):
    """Compute the largest cube edge that fits in GPU memory.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions (default 3).
    dtype_bytes : int
        Bytes per element of the working dtype (4 for float32).
    device_id : int
        CUDA/HIP device index.
    safety_factor : float
        Fraction of free VRAM to consider usable (0-1).
    n_concurrent_arrays : int
        Estimated number of dtype-sized arrays alive simultaneously.

    Returns
    -------
    int
        Largest tile edge length (rounded down to a multiple of 32).
    """
    try:
        import cupy as _cp
    except ImportError as exc:
        raise RuntimeError(
            "CuPy is required to query GPU memory"
        ) from exc

    is_rocm = _cp.cuda.runtime.is_hip
    if safety_factor == 0.85 and is_rocm:
        safety_factor = 0.70

    mem_free, mem_total = _cp.cuda.Device(device_id).mem_info
    usable = mem_free * safety_factor
    s = int((usable / (dtype_bytes * n_concurrent_arrays)) ** (1.0 / ndim))
    s = max(32, (s // 32) * 32)
    return s


def _max_chunk_size_cpu(
    ndim=3,
    dtype_bytes=4,
    safety_factor=0.70,
    n_concurrent_arrays=_PEAK_FLOAT32_ARRAYS,
):
    """Compute the largest cube edge that fits in available system RAM.

    Uses the same formula as ``max_tile_size`` but queries system memory
    via ``/proc/meminfo`` (available) instead of GPU VRAM.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions (default 3).
    dtype_bytes : int
        Bytes per element of the working dtype (4 for float32).
    safety_factor : float
        Fraction of available RAM to consider usable (0-1).
    n_concurrent_arrays : int
        Estimated number of dtype-sized arrays alive simultaneously.

    Returns
    -------
    int
        Largest chunk edge length (rounded down to a multiple of 32).
    """
    import os

    mem_available = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    usable = mem_available * safety_factor
    s = int((usable / (dtype_bytes * n_concurrent_arrays)) ** (1.0 / ndim))
    s = max(32, (s // 32) * 32)
    return s


# ToDo: Computing the chunk size should be the same for GPU and CPU, just the ammount of memory available changes. We should unify the logic and remove one of these functions.
# ToDo: Changing the chunking should come with a warning.
def _compute_chunk_size(backend_name, func_name, tile_size, device_id=0):
    """Compute memory-safe chunk size for a given function and device."""
    if _is_cupy_backend(backend_name):
        try:
            import cupy as _cp
            if _cp.cuda.runtime.is_hip:
                n_arrays = _CONCURRENT_ARRAYS_ROCM.get(
                    func_name, int(_PEAK_FLOAT32_ARRAYS * 1.5)
                )
            else:
                n_arrays = _CONCURRENT_ARRAYS.get(func_name, _PEAK_FLOAT32_ARRAYS)
            return min(
                tile_size,
                max_tile_size(ndim=3, device_id=device_id, n_concurrent_arrays=n_arrays),
            )
        except Exception:
            return tile_size

    if backend_name == "jax":
        try:
            import jax as _jax
            devs = _jax.devices()
            dev = devs[device_id] if device_id < len(devs) else devs[0]
            if dev.platform != "cpu":
                mem_stats = dev.memory_stats()
                if mem_stats:
                    mem_free = (mem_stats.get("bytes_limit", 0)
                                - mem_stats.get("bytes_in_use", 0))
                    platform = _detect_jax_platform()
                    n_arrays_map = _CONCURRENT_ARRAYS_JAX_BY_PLATFORM.get(
                        platform, _CONCURRENT_ARRAYS_JAX_CUDA,
                    )
                    n_arrays = n_arrays_map.get(
                        func_name, _PEAK_FLOAT32_ARRAYS * 2,
                    )
                    safety = 0.85
                    usable = mem_free * safety
                    s = int((usable / (4 * n_arrays)) ** (1.0 / 3))
                    s = max(32, (s // 32) * 32)
                    return min(tile_size, s)
        except Exception:
            pass

    # CPU backends: cap chunk size to available RAM
    n_arrays = _CONCURRENT_ARRAYS.get(func_name, _PEAK_FLOAT32_ARRAYS)
    return min(tile_size, _max_chunk_size_cpu(ndim=3, n_concurrent_arrays=n_arrays))


def calibrate_concurrent_arrays(
    backend_name,
    func_name,
    chunk_sizes=(64, 128, 256, 400),
    safety_factor=1.15,
    verbose=True,
):
    """Empirically measure the peak-memory multiplier for a benchmark function.

    Runs ``_prepare_inputs`` + ``_bench_function`` at each chunk size and
    reads the device's peak memory (``peak_bytes_in_use`` for JAX; computed
    from the CuPy pool for cupy/cupy-legacy).  The ratio ``peak / (chunk^3
    * 4)`` gives the effective ``n_concurrent_arrays`` factor used by
    ``_compute_chunk_size`` to pick safe chunk sizes.

    Small chunks over-report because compilation overhead dominates; the
    factor converges as the chunk grows, so the asymptotic value from the
    largest successful chunk is the useful one.

    Only one function can be calibrated per Python process for JAX because
    ``peak_bytes_in_use`` is session-monotonic — run each function in a
    fresh subprocess to calibrate all four.

    Parameters
    ----------
    backend_name : str
        ``"jax"``, ``"cupy"``, or ``"cupy-legacy"``.
    func_name : str
        ``"transform_data"``, ``"get_blending_weights"``, ``"register"``,
        or ``"fuse"``.
    chunk_sizes : sequence of int
        Ascending chunk sizes; the last successful one defines the
        asymptotic measurement.
    safety_factor : float
        Multiplier applied to the asymptote (default 1.15 = 15% headroom).
    verbose : bool
        Print per-chunk measurements as they are taken.

    Returns
    -------
    dict with keys ``measurements`` (list of per-chunk dicts),
    ``asymptotic`` (peak-over-single-array at the largest chunk),
    ``recommended`` (int, ``ceil(asymptotic * safety_factor)``), and
    ``platform`` (``"rocm"``/``"cuda"``/... for JAX, backend name otherwise).
    """
    import math

    def _peak_bytes(cs):
        if backend_name == "jax":
            import jax
            return jax.devices()[0].memory_stats().get("peak_bytes_in_use", 0)
        if _is_cupy_backend(backend_name):
            import cupy as _cp
            free, total = _cp.cuda.Device().mem_info
            return total - free
        return 0

    # Warm-up: trigger compilation and populate caches.
    try:
        warm_inputs = _prepare_inputs(
            tile_size=32, backend_name=backend_name, chunk_size=32,
        )
        _bench_function(func_name, warm_inputs, return_result=False)
    except Exception as exc:
        if verbose:
            print(f"warm-up failed: {type(exc).__name__}: {str(exc)[:120]}",
                  flush=True)

    measurements = []
    for cs in chunk_sizes:
        try:
            inputs = _prepare_inputs(
                tile_size=cs, backend_name=backend_name, chunk_size=cs,
            )
            _bench_function(func_name, inputs, return_result=False)
        except Exception as exc:
            if verbose:
                print(f"chunk={cs}: FAILED {type(exc).__name__}: "
                      f"{str(exc)[:120]}", flush=True)
            break

        peak = _peak_bytes(cs)
        single = cs ** 3 * 4
        n_total = peak / single if single else 0.0
        measurements.append({
            "chunk_size": cs,
            "peak_mb": peak / 1024 ** 2,
            "n_total": n_total,
        })
        if verbose:
            print(
                f"chunk={cs:4d}  1arr={single/1024**2:7.1f}MB  "
                f"peak={peak/1024**2:8.1f}MB  n_total={n_total:5.1f}",
                flush=True,
            )

    if not measurements:
        return {
            "measurements": [], "asymptotic": None,
            "recommended": None, "platform": None,
        }

    asymptotic = measurements[-1]["n_total"]
    recommended = int(math.ceil(asymptotic * safety_factor))

    platform = (
        _detect_jax_platform() if backend_name == "jax" else backend_name
    )
    if verbose:
        print(
            f"\n[{backend_name}/{platform}/{func_name}] asymptotic={asymptotic:.1f} "
            f"× safety={safety_factor} → recommended={recommended}",
            flush=True,
        )

    return {
        "measurements": measurements,
        "asymptotic": asymptotic,
        "recommended": recommended,
        "platform": platform,
    }


def _prepare_inputs(tile_size, backend_name, chunk_size=None, device_id=0):
    """Prepare shared inputs for benchmarking at a given tile size."""
    backend = get_backend(backend_name)
    ndim = 3

    if chunk_size is None:
        chunk_size = tile_size

    sims = sample_data.generate_tiled_dataset(
        ndim=ndim,
        N_c=1,
        N_t=1,
        tile_size=tile_size,
        tiles_x=3,
        tiles_y=3,
        tiles_z=1,
        overlap=int(tile_size * 0.2),
        spacing_x=1,
        spacing_y=1,
        spacing_z=1,
    )
    sdims = si_utils.get_spatial_dims_from_sim(sims[0])

    sim0 = sims[0].sel(
        t=sims[0].coords["t"][0],
        c=sims[0].coords["c"][0],
    ).astype(np.float32)
    sim0_chunk = sim0[{d: slice(0, chunk_size) for d in sdims}]
    sim0_data = np.asarray(sim0_chunk.data)
    sim0_data_backend = backend.asarray(sim0_data)

    return {
        "sims": sims,
        "sdims": sdims,
        "backend": backend,
        "backend_name": backend_name,
        "is_gpu": _is_gpu_backend(backend_name),
        "tile_size": tile_size,
        "chunk_size": chunk_size,
        "sim0_data_backend": sim0_data_backend,
        "p_identity": param_utils.identity_transform(ndim),
        "input_spacing": si_utils.get_spacing_from_sim(sim0_chunk, asarray=True),
        "input_origin": si_utils.get_origin_from_sim(sim0_chunk, asarray=True),
        "output_props": {
            "spacing": {d: 1.0 for d in sdims},
            "origin": {d: 0.0 for d in sdims},
            "shape": {d: chunk_size for d in sdims},
        },
        "source_bb": {
            "spacing": {d: 1.0 for d in sdims},
            "origin": {d: 0.0 for d in sdims},
            "shape": {d: chunk_size for d in sdims},
        },
        "affine": np.eye(ndim + 1),
    }


# ── Synchronisation ──────────────────────────────────────────────────


def _sync_backend(backend_name):
    """Synchronize device if needed (ensures accurate timing)."""
    if _is_cupy_backend(backend_name):
        import cupy as _cp
        _cp.cuda.Device().synchronize()
    elif backend_name == "dpnp":
        try:
            import dpctl
            dpctl.SyclQueue().wait()
        except Exception:
            pass
    elif backend_name == "jax":
        try:
            import jax
            jax.block_until_ready(jax.numpy.zeros(1))
        except Exception:
            pass


# ── Single-function benchmark ────────────────────────────────────────


def _bench_function(func_name, inputs, return_result=False):
    """Time a single invocation.

    Parameters
    ----------
    func_name : str
        One of ``"transform_data"``, ``"get_blending_weights"``,
        ``"register"``, ``"fuse"``.
    inputs : dict
        From ``_prepare_inputs()``.
    return_result : bool
        If True, also return the computed result (as numpy) under
        the ``"result"`` key.

    Returns
    -------
    dict
        Timing info and optionally the computed result.
    """
    backend = inputs["backend"]
    backend_name = inputs["backend_name"]

    _sync_backend(backend_name)

    if func_name == "transform_data":
        t0 = time.perf_counter()
        result = transformation.transform_data(
            inputs["sim0_data_backend"],
            p=inputs["p_identity"],
            input_spacing=inputs["input_spacing"],
            input_origin=inputs["input_origin"],
            output_stack_properties=inputs["output_props"],
            spatial_dims=inputs["sdims"],
            backend=backend,
            order=1,
            cval=0.0,
        )
        _sync_backend(backend_name)
        out = {"compute_time": time.perf_counter() - t0}
        if return_result:
            out["result"] = np.asarray(backend.to_numpy(result))
        return out

    elif func_name == "get_blending_weights":
        t0 = time.perf_counter()
        result = weights.get_blending_weights(
            target_bb=inputs["output_props"],
            source_bb=inputs["source_bb"],
            affine=inputs["affine"],
            backend=backend_name,
        )
        _sync_backend(backend_name)
        out = {"compute_time": time.perf_counter() - t0}
        if return_result:
            out["result"] = np.asarray(backend.to_numpy(result))
        return out

    elif func_name == "register":
        msims = [
            msi_utils.get_msim_from_sim(s, scale_factors=[])
            for s in inputs["sims"]
        ]
        t0 = time.perf_counter()
        registration.register(
            msims,
            reg_channel_index=0,
            transform_key=METADATA_TRANSFORM_KEY,
            backend=backend_name,
        )
        _sync_backend(backend_name)
        out = {"compute_time": time.perf_counter() - t0}
        if return_result:
            # Extract affine matrices from the registered msims
            sims_reg = [msi_utils.get_sim_from_msim(m) for m in msims]
            affines = [
                np.asarray(
                    si_utils.get_affine_from_sim(
                        s, transform_key=METADATA_TRANSFORM_KEY
                    )
                )
                for s in sims_reg
            ]
            out["result"] = affines
        return out

    elif func_name == "fuse":
        _chunk_times = []
        _orig = fusion.fuse_np

        def _timed_fuse_np(*a, **kw):
            _sync_backend(backend_name)
            _t0 = time.perf_counter()
            r = _orig(*a, **kw)
            _sync_backend(backend_name)
            _chunk_times.append(time.perf_counter() - _t0)
            if _is_cupy_backend(backend_name):
                r = backend.to_numpy(r)
                _free_gpu_pool()
            return r

        fusion.fuse_np = _timed_fuse_np
        sched = backend.recommended_dask_scheduler or "single-threaded"
        chunksize = {d: inputs["chunk_size"] for d in inputs["sdims"]}

        t0_total = time.perf_counter()
        with dask.config.set(scheduler=sched):
            fused = fusion.fuse(
                inputs["sims"],
                transform_key=METADATA_TRANSFORM_KEY,
                output_chunksize=chunksize,
                backend=backend_name,
            ).compute()
        total_time = time.perf_counter() - t0_total
        fusion.fuse_np = _orig

        _free_gpu_pool()

        chunk_total = sum(_chunk_times)
        out = {
            "compute_time": chunk_total,
            "total_time": total_time,
            "dask_overhead": total_time - chunk_total,
            "n_chunks": len(_chunk_times),
        }
        if return_result:
            out["result"] = np.asarray(fused.values)
        return out

    else:
        raise ValueError(f"Unknown function: {func_name}")


# ── Correctness comparison ───────────────────────────────────────────

_CORRECTNESS_ATOL = {
    "transform_data": 1e-3,
    "get_blending_weights": 1e-3,
    "register": 1e-3,
    "fuse": 1.0,
}


def _compare_results(func_name, result, reference):
    """Compare a benchmark result to a numpy reference.

    Returns
    -------
    dict
        ``{"max_abs_diff": float, "correct": bool}``
    """
    atol = _CORRECTNESS_ATOL.get(func_name, 1e-3)

    if func_name == "register":
        # Compare lists of affine matrices
        diffs = []
        for r, ref in zip(result, reference):
            diffs.append(float(np.abs(np.asarray(r).astype(float)
                                      - np.asarray(ref).astype(float)).max()))
        max_diff = max(diffs) if diffs else 0.0
    else:
        max_diff = float(
            np.abs(
                np.asarray(result).astype(float)
                - np.asarray(reference).astype(float)
            ).max()
        )

    return {
        "max_abs_diff": max_diff,
        "correct": max_diff <= atol,
    }


# ── Atomic JSON writer ───────────────────────────────────────────────


def _write_results_json(result_dict, output_path):
    """Write result dict to JSON atomically."""
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(output_path),
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(result_dict, f, indent=2)
        os.replace(tmp_path, output_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ── Phase 1: Plan ────────────────────────────────────────────────────

DEFAULT_FUNCTIONS = [
    "transform_data",
    "get_blending_weights",
    "register",
    "fuse",
]


def plan_benchmarks(
    specs=("numpy",),
    tile_sizes=(100,),
    functions=DEFAULT_FUNCTIONS,
    numba_acceleration=None,
):
    """Plan which benchmarks to run.

    Parameters
    ----------
    specs : sequence of str
        Backend specs in ``"backend_name[:device_type[:device_id]]"``
        format.  See module docstring for examples.
    tile_sizes : sequence of int
        Tile edge lengths for 3D benchmarks.
    functions : sequence of str
        Which functions to benchmark.
    numba_acceleration : bool, list of bool, or None
        Controls numba acceleration during benchmarks.
        ``[True, False]`` runs each job twice (with and without numba).
        ``True``/``False`` runs only that variant.
        ``None`` uses the current global setting (single run).

    Returns
    -------
    dict
        Plan with ``"jobs"``, ``"groups"``, ``"system_info"``, ``"config"``.
    """
    # Normalise numba_acceleration to a list
    if numba_acceleration is None:
        numba_variants = [None]
    elif isinstance(numba_acceleration, bool):
        numba_variants = [numba_acceleration]
    else:
        numba_variants = list(numba_acceleration)

    # Parse specs and enumerate devices
    jobs = []
    groups_seen = []
    skipped = []

    for spec in specs:
        try:
            parsed = parse_backend_spec(spec)
        except ValueError as e:
            skipped.append((spec, str(e)))
            continue

        backend_name = parsed["backend_name"]
        devices = _enumerate_devices(
            backend_name, parsed["device_type"], parsed["device_id"],
        )

        if not devices:
            skipped.append((spec, "no matching devices found"))
            continue

        for dev_id, dev_label in devices:
            group_key = f"{backend_name}:{dev_label}"
            if group_key not in groups_seen:
                groups_seen.append(group_key)

            for tile_size in tile_sizes:
                for func_name in functions:
                    chunk_size = _compute_chunk_size(
                        backend_name, func_name, tile_size, dev_id,
                    )
                    for numba_val in numba_variants:
                        jobs.append({
                            "backend_name": backend_name,
                            "device_id": dev_id,
                            "device_label": dev_label,
                            "tile_size": tile_size,
                            "function": func_name,
                            "chunk_size": chunk_size,
                            "group_key": group_key,
                            "numba_enabled": numba_val,
                        })

    system_info = _gather_system_info()

    plan = {
        "jobs": jobs,
        "groups": groups_seen,
        "system_info": system_info,
        "config": {
            "specs": list(specs),
            "tile_sizes": list(tile_sizes),
            "functions": list(functions),
            "numba_acceleration": numba_variants,
        },
    }

    if skipped:
        print(f"Skipped: {skipped}")

    # Print summary
    print(f"\nBenchmark plan: {len(jobs)} jobs across {len(groups_seen)} groups")
    print(f"  Tile sizes: {list(tile_sizes)}")
    print(f"  Functions:  {list(functions)}")
    if any(v is not None for v in numba_variants):
        print(f"  Numba:      {numba_variants}")
    print(f"  Groups:")
    for g in groups_seen:
        n = sum(1 for j in jobs if j["group_key"] == g)
        print(f"    {g}: {n} jobs")
    print()

    return plan


# ── Phase 2: Run ─────────────────────────────────────────────────────


def run_benchmarks(
    plan,
    output_path=None,
    n_runs=1,
    warm_up_runs=1,
    max_warm_up_time=60,
    check_correctness=False,
):
    """Execute a benchmark plan.

    Parameters
    ----------
    plan : dict
        From ``plan_benchmarks()``.
    output_path : str, os.PathLike, or None
        Directory or file path for JSON results.  If a directory (or a
        path without a ``.json`` extension), a timestamped filename is
        generated inside it (e.g. ``benchmark_2026-03-26_14-30-05.json``).
        Written incrementally after each group completes.
        If None, results are not saved to disk.
    n_runs : int
        Timed repetitions per job.
    warm_up_runs : int
        Warm-up iterations per job.
    max_warm_up_time : float
        Seconds after which remaining warm-up iterations are skipped.
    check_correctness : bool
        If True, compare each run's result against a numpy reference
        and log per-iteration correctness data.

    Returns
    -------
    dict
        JSON-serializable result dictionary.
    """
    # Resolve output_path: directory -> timestamped file inside it
    if output_path is not None:
        output_path = str(output_path)
        if os.path.isdir(output_path) or not output_path.endswith(".json"):
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_path = os.path.join(output_path, f"benchmark_{ts}.json")

    system_info = plan["system_info"]
    config = dict(plan["config"])
    config.update({
        "n_runs": n_runs,
        "warm_up_runs": warm_up_runs,
        "max_warm_up_time": max_warm_up_time,
        "check_correctness": check_correctness,
    })

    result_dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "comment": "",
        "system_info": system_info,
        "config": config,
        "results": [],
    }

    jobs = plan["jobs"]
    groups = plan["groups"]

    # Correctness references: computed lazily, keyed by (tile_size, function, numba_enabled)
    references = {}

    # Save and restore numba acceleration state
    from multiview_stitcher._numba_acceleration import (
        _use_numba_acceleration as _orig_numba,
    )

    try:
        for group_key in groups:
            group_jobs = [j for j in jobs if j["group_key"] == group_key]
            if not group_jobs:
                continue

            backend_name = group_jobs[0]["backend_name"]
            device_id = group_jobs[0]["device_id"]

            # Release GPU memory from previous group so the next backend
            # starts with a clean device (e.g. CuPy pool before JAX).
            _free_gpu_pool()
            import gc as _gc_group
            _gc_group.collect()

            # Set up device context
            _jax_ctx = None
            if _is_cupy_backend(backend_name):
                import cupy as _cp
                _cp.cuda.Device(device_id).use()
                if _cp.cuda.runtime.is_hip:
                    n_cupy_devices = _cp.cuda.runtime.getDeviceCount()
                    if n_cupy_devices > 1:
                        import shutil
                        shutil.rmtree(
                            _cp.cuda.compiler.get_cache_dir(),
                            ignore_errors=True,
                        )
                        _dev_cache = os.path.join(
                            os.path.expanduser("~"),
                            ".cupy",
                            f"kernel_cache_gpu{device_id}",
                        )
                        os.makedirs(_dev_cache, exist_ok=True)
                        os.environ["CUPY_CACHE_DIR"] = _dev_cache
                        _cp.cuda.compiler._empty_file_preprocess_cache.clear()
            elif backend_name == "jax":
                try:
                    import jax as _jax
                    jax_devices = _jax.devices()
                    if device_id < len(jax_devices):
                        _jax_ctx = _jax.default_device(jax_devices[device_id])
                        _jax_ctx.__enter__()
                    # Enable fp64 only on devices with native support.
                    # TPU lacks fp64/complex128 hardware — enabling x64
                    # would crash FFT-based registration (C128).
                    if _jax.default_backend() != "tpu":
                        _jax.config.update("jax_enable_x64", True)
                    else:
                        print("[TPU — skipping jax_enable_x64] ",
                              end="")
                except Exception:
                    pass

            try:
                for job in group_jobs:
                    fn = job["function"]
                    tile_size = job["tile_size"]
                    numba_val = job["numba_enabled"]

                    # Recompute chunk size at run time so it reflects
                    # actual memory availability (plan-time estimates
                    # can be stale, e.g. after a CuPy group).
                    chunk_size = _compute_chunk_size(
                        backend_name, fn, tile_size, device_id,
                    )

                    # Set numba acceleration
                    if numba_val is not None:
                        multiview_stitcher.set_numba_acceleration(numba_val)

                    numba_tag = ""
                    if numba_val is not None:
                        numba_tag = f"/numba={'on' if numba_val else 'off'}"

                    tag = f"{group_key}/tile={tile_size}{numba_tag}"
                    label = f"  {tag}: {fn} [chunk={chunk_size}]"
                    print(f"\n{label}: ", end="", flush=True)

                    # Release previous arrays and collect garbage
                    try:
                        del inputs
                    except NameError:
                        pass
                    import gc as _gc
                    _gc.collect()
                    if _is_cupy_backend(backend_name):
                        _free_gpu_pool()

                    inputs = _prepare_inputs(
                        tile_size, backend_name,
                        chunk_size=chunk_size,
                        device_id=device_id,
                    )

                    # Compute numpy reference if needed (lazy)
                    ref_key = (tile_size, fn, numba_val)
                    if check_correctness and ref_key not in references:
                        if numba_val is not None:
                            multiview_stitcher.set_numba_acceleration(numba_val)
                        ref_inputs = _prepare_inputs(
                            tile_size, "numpy", chunk_size=chunk_size,
                        )
                        ref_timing = _bench_function(fn, ref_inputs, return_result=True)
                        references[ref_key] = ref_timing["result"]
                        del ref_inputs
                        _gc.collect()

                    # Warm-up
                    warm_t0 = time.perf_counter()
                    for w in range(warm_up_runs):
                        if time.perf_counter() - warm_t0 > max_warm_up_time:
                            print(f"[warm-up capped at {w}/{warm_up_runs}] ", end="")
                            break
                        _bench_function(fn, inputs)
                    if _is_cupy_backend(backend_name):
                        _free_gpu_pool()

                    # Memory before timed runs
                    mem_before = _get_memory_usage_mb(backend_name)
                    print(f"[mem_before={_fmt_mem(*mem_before)}] ", end="", flush=True)

                    # Timed runs
                    timings = []
                    correctness_results = []
                    for _ in range(n_runs):
                        t = _bench_function(fn, inputs, return_result=check_correctness)
                        timings.append(t)
                        if check_correctness and "result" in t:
                            ref = references.get(ref_key)
                            if ref is not None:
                                corr = _compare_results(fn, t["result"], ref)
                                correctness_results.append(corr)
                            # Free the result to save memory
                            del t["result"]

                    # Memory after timed runs
                    mem_after = _get_memory_usage_mb(backend_name)
                    print(f"[mem_after={_fmt_mem(*mem_after)}] ", end="", flush=True)

                    # Free reference array once all runs for this key are done
                    if ref_key in references:
                        del references[ref_key]
                        _gc.collect()

                    compute_times = [t["compute_time"] for t in timings]

                    record = {
                        "function": fn,
                        "backend": backend_name,
                        "device": job["device_label"],
                        "tile_size": tile_size,
                        "chunk_size": chunk_size,
                        "numba_enabled": numba_val,
                        "times_s": compute_times,
                        "mean_time_s": float(np.mean(compute_times)),
                        "std_time_s": float(np.std(compute_times)),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "mem_before_mb": mem_before[0],
                        "mem_after_mb": mem_after[0],
                        "mem_total_mb": mem_before[1],
                    }

                    if fn == "fuse":
                        record["dask_overhead_s"] = float(
                            np.mean([t["dask_overhead"] for t in timings])
                        )
                        record["total_time_s"] = float(
                            np.mean([t["total_time"] for t in timings])
                        )
                        record["n_chunks"] = timings[0]["n_chunks"]

                    if correctness_results:
                        record["correctness"] = correctness_results

                    result_dict["results"].append(record)
                    print(f"{record['mean_time_s']:.3f}s", end="")

                    if correctness_results:
                        all_correct = all(c["correct"] for c in correctness_results)
                        max_diff = max(c["max_abs_diff"] for c in correctness_results)
                        symbol = "ok" if all_correct else "FAIL"
                        print(f" [{symbol}, diff={max_diff:.2e}]", end="")

            except Exception as exc:
                print(
                    f"\n  [error in {group_key}: "
                    f"{type(exc).__name__}: {exc}]"
                )
            finally:
                if _jax_ctx is not None:
                    _jax_ctx.__exit__(None, None, None)

            # Incremental write after each group
            if output_path is not None:
                _write_results_json(result_dict, output_path)
                print(f"\n  [results saved to {output_path}]")

    finally:
        # Restore original numba state
        multiview_stitcher.set_numba_acceleration(_orig_numba)

    # Final write
    if output_path is not None:
        _write_results_json(result_dict, output_path)
        print(f"\nResults saved to {output_path}")

    n = len(result_dict["results"])
    sid = system_info["id"]
    print(f"\nComplete: {n} measurements (System ID: {sid})")

    return result_dict


# ── Phase 3: Evaluate & Plot ─────────────────────────────────────────


def _resolve_device_name(system_info, device_label):
    """Map a device label like ``"gpu:0"`` to the actual hardware name."""
    if device_label.startswith("gpu:"):
        idx = int(device_label.split(":")[1])
        gpus = system_info.get("gpus", [])
        if idx < len(gpus):
            return gpus[idx]["name"]
    elif device_label.startswith("xpu:"):
        idx = int(device_label.split(":")[1])
        gpus = system_info.get("gpus", [])
        if idx < len(gpus):
            return gpus[idx]["name"]
        return device_label
    elif device_label.startswith("cpu"):
        return system_info.get("cpu", "CPU")
    elif device_label.startswith("device:"):
        gpus = system_info.get("gpus", [])
        if gpus:
            return gpus[0]["name"]
        return system_info.get("cpu", "CPU")
    return device_label


def _short_device_name(name):
    """Shorten a verbose hardware name for plot labels."""
    import re

    s = name
    for prefix in ("NVIDIA GeForce ", "NVIDIA ", "AMD Instinct ", "AMD "):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    s = s.replace("(R)", "")
    s = re.sub(r"\s*CPU\s*@.*", "", s)
    s = " ".join(s.split())
    return s


def evaluate_benchmarks(results):
    """Load benchmark results, print a discovery summary, return merged data.

    Parameters
    ----------
    results : dict, str, os.PathLike, or list thereof
        A single benchmark result (dict or path to JSON), or a list of
        them.  When a path points to a **directory**, every
        ``*.json`` file inside it is loaded.

    Returns
    -------
    dict
        Merged data with keys: ``"records"``, ``"systems"``,
        ``"functions"``, ``"tile_sizes"``, ``"backends"``, ``"devices"``.
    """
    import glob as _glob

    if not isinstance(results, list):
        results = [results]

    raw_dicts = []
    for item in results:
        if isinstance(item, dict):
            raw_dicts.append(item)
        elif isinstance(item, (str, os.PathLike)):
            p = str(item)
            if os.path.isdir(p):
                for fp in sorted(
                    _glob.glob(os.path.join(p, "*.json"))
                ):
                    with open(fp) as f:
                        raw_dicts.append(json.load(f))
            else:
                with open(p) as f:
                    raw_dicts.append(json.load(f))

    if not raw_dicts:
        raise ValueError("No benchmark results found.")

    # Unify system IDs for the same physical hardware
    import hashlib as _hl

    _sid_map = {}
    for d in raw_dicts:
        sinfo = d["system_info"]
        old_sid = sinfo["id"]
        fp_parts = [sinfo.get("cpu", "unknown"),
                     str(sinfo.get("ram_gb", ""))]
        canonical = _hl.sha256("|".join(fp_parts).encode()).hexdigest()[:12]
        _sid_map[old_sid] = canonical

    # Merge
    all_records = []
    systems = {}
    all_functions = set()
    all_tile_sizes = set()
    all_backends = set()
    all_devices = set()

    for d in raw_dicts:
        sinfo = d["system_info"]
        sid = _sid_map[sinfo["id"]]
        canonical_info = dict(sinfo, id=sid)
        if sid in systems:
            existing_names = {g.get("name") for g in systems[sid].get("gpus", [])}
            for g in canonical_info.get("gpus", []):
                if g.get("name") not in existing_names:
                    systems[sid]["gpus"].append(g)
                    existing_names.add(g.get("name"))
        else:
            systems[sid] = canonical_info
        for r in d["results"]:
            device_name = _resolve_device_name(canonical_info, r["device"])
            rec = dict(
                r,
                system_id=sid,
                n_samples=len(r["times_s"]),
                device_name=device_name,
            )
            all_records.append(rec)
            all_functions.add(r["function"])
            all_tile_sizes.add(r["tile_size"])
            all_backends.add(r["backend"])
            all_devices.add(device_name)

    merged = {
        "records": all_records,
        "systems": systems,
        "functions": sorted(all_functions),
        "tile_sizes": sorted(all_tile_sizes),
        "backends": sorted(all_backends),
        "devices": sorted(all_devices),
    }

    # Discovery summary
    print(f"Loaded {len(raw_dicts)} benchmark file(s)\n")

    print("Systems:")
    for sid, si in systems.items():
        cpu = si.get("cpu", "unknown")
        gpus = si.get("gpus", [])
        def _gpu_label(g):
            name = g.get("name", "unknown")
            mem = g.get("memory_total_gb")
            if mem is not None:
                return f"{name} ({mem} GB)"
            return name

        gpu_str = (
            ", ".join(_gpu_label(g) for g in gpus)
            if gpus
            else "none"
        )
        print(f"  {sid}  CPU: {cpu}")
        print(f"  {'':>{len(sid)}}  GPU: {gpu_str}")

    print(f"\nFunctions:  {', '.join(merged['functions'])}")
    print(f"Tile sizes: {merged['tile_sizes']}")
    print(f"Backends:   {', '.join(merged['backends'])}")
    print(f"Devices:    {', '.join(merged['devices'])}")

    # Per-combination summary table
    print(f"\nRecords: {len(all_records)} total")
    combos = {}
    for r in all_records:
        numba = r.get("numba_enabled")
        numba_str = (
            "on" if numba is True
            else "off" if numba is False
            else "-"
        )
        key = (r["backend"], r["device_name"], numba_str, r["function"],
               r["tile_size"])
        if key not in combos:
            combos[key] = 0
        combos[key] += r["n_samples"]

    hdr_dev = "device"
    max_dev = max(len(hdr_dev), max(len(k[1]) for k in combos))
    max_be = max(7, max(len(k[0]) for k in combos))
    max_fn = max(8, max(len(k[3]) for k in combos))
    print(
        f"  {'backend':<{max_be}s} {hdr_dev:<{max_dev}s} {'numba':>5s} "
        f"{'function':<{max_fn}s} {'tile':>6s} {'samples':>8s}"
    )
    print(
        f"  {'-'*max_be} {'-'*max_dev} {'-'*5} "
        f"{'-'*max_fn} {'-'*6} {'-'*8}"
    )
    for (b, d, nb, fn, ts), n_samples in sorted(combos.items()):
        print(
            f"  {b:<{max_be}s} {d:<{max_dev}s} {nb:>5s} "
            f"{fn:<{max_fn}s} {ts:>6d} {n_samples:>8d}"
        )

    return merged


def plot_runtime_and_speedup(merged):
    """Plot benchmark results: linear, log, and per-device views.

    For each benchmarked function, three types of figures are produced:

    1. **Linear (all devices)** — bars on a linear y-axis, all devices
       side by side.
    2. **Logarithmic (all devices)** — same layout but with a log y-axis
       so that fast GPU bars are distinguishable from each other even when
       CPU bars are orders of magnitude larger.
    3. **Per-device linear** — one figure per device, linear y-axis,
       showing only the backends that ran on that device.

    When benchmark data includes both ``numba_enabled=True`` and
    ``numba_enabled=False`` records for the same backend, each numba
    variant gets its own bar.  Numba-off bars use a hatched pattern
    of the same colour as the numba-on bar.

    Parameters
    ----------
    merged : dict
        Value returned by ``evaluate_benchmarks()``.

    Returns
    -------
    list of matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from itertools import groupby

    records = merged["records"]
    systems = merged["systems"]
    functions = merged["functions"]
    devices = merged["devices"]
    tile_sizes = merged["tile_sizes"]
    backends = merged["backends"]

    # ── Numba series logic ───────────────────────────────────────────
    numba_values = sorted(
        {r.get("numba_enabled") for r in records},
        key=lambda v: (v is None, v is False, v is True),
    )
    has_numba_split = True in numba_values and False in numba_values

    if has_numba_split:
        series_list = []
        for b in backends:
            # Check if this backend has numba True/False variants
            has_true = any(r["backend"] == b and r.get("numba_enabled") is True
                          for r in records)
            has_false = any(r["backend"] == b and r.get("numba_enabled") is False
                           for r in records)
            if has_true or has_false:
                for nb in [True, False]:
                    if any(r["backend"] == b and r.get("numba_enabled") == nb
                           for r in records):
                        series_list.append((b, nb))
            else:
                # Backend has only numba_enabled=None (e.g. "original")
                series_list.append((b, None))
    else:
        series_list = [(b, None) for b in backends]

    # ── Helpers ──────────────────────────────────────────────────────

    def _get_records(func, backend, numba_enabled, device_name,
                     tile_size, system_id=None):
        out = []
        for r in records:
            if r["function"] != func:
                continue
            if r["backend"] != backend:
                continue
            if r["device_name"] != device_name:
                continue
            if r["tile_size"] != tile_size:
                continue
            if system_id is not None and r["system_id"] != system_id:
                continue
            if numba_enabled is not None:
                if r.get("numba_enabled") != numba_enabled:
                    continue
            out.append(r)
        return out

    def _aggregate(recs):
        all_times = []
        for r in recs:
            all_times.extend(r["times_s"])
        if not all_times:
            return None
        return {
            "mean": float(np.mean(all_times)),
            "std": float(np.std(all_times)),
            "n": len(all_times),
        }

    _PALETTE = [
        "#4C72B0", "#C44E52", "#55A868", "#8172B2", "#CCB974", "#64B5CD",
    ]
    _FIXED_COLORS = {"original": "#000000"}
    backend_color = {}
    palette_idx = 0
    for b in backends:
        if b in _FIXED_COLORS:
            backend_color[b] = _FIXED_COLORS[b]
        else:
            backend_color[b] = _PALETTE[palette_idx % len(_PALETTE)]
            palette_idx += 1

    short_names = {dev: _short_device_name(dev) for dev in devices}

    _BASELINE_BACKENDS = ["original", "numpy", "numpy-legacy"]

    def _baseline_for_system(func, tile_size, system_id):
        for ref_backend in _BASELINE_BACKENDS:
            for nb in ([True, None] if has_numba_split else [None]):
                ref_recs = [
                    r for r in records
                    if r["function"] == func
                    and r["backend"] == ref_backend
                    and r["tile_size"] == tile_size
                    and r["system_id"] == system_id
                    and (nb is None or r.get("numba_enabled") == nb)
                ]
                agg = _aggregate(ref_recs)
                if agg:
                    return agg, ref_backend, ref_recs[0]["device_name"]

        my_cpu = systems.get(system_id, {}).get("cpu")
        if my_cpu:
            sibling_sids = [
                sid for sid, sinfo in systems.items()
                if sid != system_id and sinfo.get("cpu") == my_cpu
            ]
            for sib_sid in sibling_sids:
                for ref_backend in _BASELINE_BACKENDS:
                    for nb in ([True, None] if has_numba_split
                               else [None]):
                        ref_recs = [
                            r for r in records
                            if r["function"] == func
                            and r["backend"] == ref_backend
                            and r["tile_size"] == tile_size
                            and r["system_id"] == sib_sid
                            and (nb is None
                                 or r.get("numba_enabled") == nb)
                        ]
                        agg = _aggregate(ref_recs)
                        if agg:
                            return (agg, ref_backend,
                                    ref_recs[0]["device_name"])

        return None, None, None

    GROUP_GAP = 1.0

    all_sd_pairs = sorted(
        {(r["system_id"], r["device_name"]) for r in records},
        key=lambda sd: (sd[1], sd[0]),
    )
    device_counts = {}
    for _, dev in all_sd_pairs:
        device_counts[dev] = device_counts.get(dev, 0) + 1

    # ── Slot builder ─────────────────────────────────────────────────

    def _build_slots(func, sd_pairs):
        """Build bar slots and device spans for the given device pairs."""
        slots = []
        device_spans = []
        x_pos = 0.0

        for sid, dev in sd_pairs:
            dev_start = None

            for ts in tile_sizes:
                slot_series = [
                    (b, nb) for b, nb in series_list
                    if _aggregate(_get_records(
                        func, b, nb, dev, ts, sid
                    )) is not None
                ]
                if not slot_series:
                    continue

                if dev_start is None:
                    dev_start = x_pos

                n_s = len(slot_series)
                bar_w = 0.7 / max(n_s, 1)

                for si, (backend, numba_en) in enumerate(slot_series):
                    agg = _aggregate(
                        _get_records(func, backend, numba_en,
                                     dev, ts, sid)
                    )
                    cx = x_pos + (-0.35 + bar_w * (si + 0.5))
                    slots.append({
                        "x": cx,
                        "width": bar_w,
                        "device": dev,
                        "system_id": sid,
                        "tile_size": ts,
                        "backend": backend,
                        "numba_enabled": numba_en,
                        "agg": agg,
                        "group_center": x_pos,
                    })

                x_pos += 1.0

            if dev_start is not None:
                device_spans.append(
                    (dev_start, x_pos - 1.0, dev, sid)
                )
                x_pos += GROUP_GAP

        return slots, device_spans

    # ── Axes painter ─────────────────────────────────────────────────

    def _paint_axes(ax, func, slots, device_spans, log_scale=False):
        """Draw bars, labels, annotations onto *ax*."""
        legend_seen = set()

        for s in slots:
            backend = s["backend"]
            numba_en = s["numba_enabled"]
            color = backend_color[backend]

            if has_numba_split and numba_en is not None:
                legend_key = (backend, numba_en)
                label_text = (
                    f"{backend} (numba)"
                    if numba_en
                    else f"{backend} (no numba)"
                )
            else:
                legend_key = (backend, None)
                label_text = backend

            label = (label_text if legend_key not in legend_seen
                     else None)
            legend_seen.add(legend_key)

            hatch = "//" if numba_en is False else None
            face_color = color if numba_en is not False else "white"
            edge_color = color

            ax.bar(
                s["x"], s["agg"]["mean"], s["width"],
                yerr=s["agg"]["std"], capsize=3,
                color=face_color, edgecolor=edge_color,
                hatch=hatch, linewidth=0.8,
                label=label,
            )

            # Value label above bar
            if log_scale:
                label_y = s["agg"]["mean"] * 1.15
            else:
                label_y = s["agg"]["mean"] + s["agg"]["std"]
            ax.text(
                s["x"], label_y,
                f"{s['agg']['mean']:.3f}s",
                ha="center", va="bottom", fontsize=8,
            )

        if log_scale:
            ax.set_yscale("log")

        ax.autoscale_view()

        _baseline_cache = {}
        _ref_descriptions = set()

        for (sid, dev, ts), grp in groupby(
            slots, key=lambda s: (
                s["system_id"], s["device"], s["tile_size"],
            )
        ):
            grp = list(grp)
            center = grp[0]["group_center"]

            ax.text(
                center, -0.02,
                f"tile={ts}",
                ha="center", va="top", fontsize=8,
                transform=ax.get_xaxis_transform(),
            )

            baseline_mean = None
            if (sid, ts) not in _baseline_cache:
                _baseline_cache[(sid, ts)] = \
                    _baseline_for_system(func, ts, sid)
            bl_agg, bl_backend, bl_dev = _baseline_cache[(sid, ts)]
            if bl_agg:
                baseline_mean = bl_agg["mean"]
                _ref_descriptions.add(
                    f"{bl_backend} on {short_names[bl_dev]}"
                )

            for s in grp:
                agg = s["agg"]
                lines = []
                if baseline_mean and baseline_mean > 0:
                    lines.append(
                        f"{baseline_mean / agg['mean']:.1f}x"
                    )
                lines.append(f"n={agg['n']}")
                if agg["n"] > 1:
                    lines.append(f"\u03c3={agg['std']:.3f}")
                if lines:
                    ax.text(
                        s["x"], -0.07,
                        "\n".join(lines),
                        ha="center", va="top", fontsize=7,
                        color="0.35", style="italic",
                        transform=ax.get_xaxis_transform(),
                    )

        for x_start, x_end, dev, sid in device_spans:
            cx = (x_start + x_end) / 2
            if device_counts.get(dev, 0) > 1:
                dev_label = f"{short_names[dev]} ({sid})"
            else:
                dev_label = short_names[dev]
            ax.text(
                cx, -0.22,
                dev_label,
                ha="center", va="top", fontsize=10,
                fontweight="bold",
                transform=ax.get_xaxis_transform(),
            )

        ax.set_xticks([])
        ax.set_ylabel("Time (s)")

        if _ref_descriptions:
            ref_str = ", ".join(sorted(_ref_descriptions))
            subtitle = f"Speedup relative to {ref_str}"
        else:
            subtitle = ""

        return subtitle

    def _set_title(ax, func, subtitle, suffix=""):
        title = f"Backend Benchmark: {func}"
        if suffix:
            title += f" ({suffix})"
        ax.set_title(title, fontsize=12, fontweight="bold",
                     pad=16 if subtitle else 6)
        if subtitle:
            ax.text(
                0.5, 1.01, subtitle,
                ha="center", va="bottom", fontsize=8,
                color="0.4", style="italic",
                transform=ax.transAxes,
            )

    # ── Generate figures ─────────────────────────────────────────────

    figures = []

    for func in functions:
        # --- 1. Linear, all devices ---
        slots, spans = _build_slots(func, all_sd_pairs)
        if not slots:
            continue

        fig, ax = plt.subplots(
            figsize=(max(5, 1.4 * len(slots) + 1.5), 5),
        )
        subtitle = _paint_axes(ax, func, slots, spans, log_scale=False)
        _set_title(ax, func, subtitle)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3, axis="y")
        fig.subplots_adjust(bottom=0.28, top=0.88)
        figures.append(fig)

        # --- 2. Log, all devices ---
        fig_log, ax_log = plt.subplots(
            figsize=(max(5, 1.4 * len(slots) + 1.5), 5),
        )
        subtitle = _paint_axes(ax_log, func, slots, spans,
                               log_scale=True)
        _set_title(ax_log, func, subtitle, suffix="log scale")
        ax_log.legend(fontsize=9, loc="best")
        ax_log.grid(True, alpha=0.3, axis="y", which="both")
        fig_log.subplots_adjust(bottom=0.28, top=0.88)
        figures.append(fig_log)

        # --- 3. Per-device, linear ---
        unique_devices = []
        seen_devs = set()
        for sid, dev in all_sd_pairs:
            if dev not in seen_devs:
                seen_devs.add(dev)
                unique_devices.append(dev)

        if len(unique_devices) > 1:
            for target_dev in unique_devices:
                sd_subset = [
                    (sid, dev) for sid, dev in all_sd_pairs
                    if dev == target_dev
                ]
                dev_slots, dev_spans = _build_slots(func, sd_subset)
                if not dev_slots:
                    continue

                fig_d, ax_d = plt.subplots(
                    figsize=(max(5, 1.4 * len(dev_slots) + 1.5), 5),
                )
                subtitle = _paint_axes(ax_d, func, dev_slots,
                                       dev_spans, log_scale=False)
                _set_title(ax_d, func, subtitle,
                           suffix=short_names[target_dev])
                ax_d.legend(fontsize=9, loc="best")
                ax_d.grid(True, alpha=0.3, axis="y")
                fig_d.subplots_adjust(bottom=0.28, top=0.88)
                figures.append(fig_d)

    plt.show()
    return figures
