"""Abstract backend interface for multiview-stitcher.

``Backend`` is a minimal marker base class.  The actual default
implementations live in ``XPBackend`` (for xp-based backends) or are
provided directly by legacy backends.

The only concrete defaults here are ``free_memory`` (no-op) and
``recommended_dask_scheduler`` (``None``), which are sensible for any
CPU backend.
"""


class Backend:
    """Base class for all array backends.

    Subclass either this (for fully custom backends) or ``XPBackend``
    (to get automatic delegation to an array module).
    """

    def free_memory(self):
        """Free cached memory (no-op for CPU backends)."""
        pass

    @property
    def recommended_dask_scheduler(self):
        """Return recommended dask scheduler, or None for dask default."""
        return None

    def __repr__(self):
        return f"{self.__class__.__name__}()"
