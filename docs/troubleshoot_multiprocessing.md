# RuntimeError: multiprocessing freeze_support

## Symptom

When running a script that calls `register()` or other functions using `scheduler="processes"`, you encounter:

```
RuntimeError:
    An attempt has been made to start a new process before the
    current process has finished its bootstrapping phase.
    ...
        if __name__ == '__main__':
            freeze_support()
            ...
```

## Cause

On macOS and Windows, Python uses the `spawn` multiprocessing start method by default. When spawning worker processes, Python re-imports your script as `__main__`, which re-executes all top-level code — including the `register()` call itself — before the worker is ready.

## Fix

Wrap all executable code in your script with the `if __name__ == '__main__':` guard:

```python
from multiview_stitcher import registration, fusion

# Setup (safe at module level)
msims = [...]

if __name__ == '__main__':
    params = registration.register(msims, ...)
    fused = fusion.fuse(...)
```
