try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from multiview_stitcher._numba_acceleration import (
    get_numba_acceleration,
    numba_available,
    set_numba_acceleration,
)
from multiview_stitcher.backends import get_backend, set_backend

__all__ = (
    "get_backend",
    "set_backend",
    "get_numba_acceleration",
    "set_numba_acceleration",
    "numba_available",
)
