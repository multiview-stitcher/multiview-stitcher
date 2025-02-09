import logging
import warnings
from contextlib import contextmanager

import dask.array as da
import numpy as np
from dask import delayed


class DisableLogger:
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


@contextmanager
def temporary_log_level(logger, level):
    """
    Use in notebooks:

    import logging, sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout,
        force=True)

    with temporary_log_level(logging.getLogger('multiview_stitcher'), logging.DEBUG):
        ...

    """
    old_level = logger.level
    logger.setLevel(level)
    yield
    logger.setLevel(old_level)


def get_zarr_array_from_dask_array(dask_array):
    """
    If `dask_array` was created using
      - `da.from_zarr` or
      - `da.to_zarr(return_stored=True,...),

    return the underlying zarr array.

    Otherwise, return None.

    Note: If operations have been applied after `da.from_zarr`, this function returns None.
    """
    # check for from_zarr
    keys = list(dask_array.dask.keys())
    if not (
        keys[0][0].startswith("from-zarr")
        or keys[0][0].startswith("load-store-map")
    ):
        # warn that the dask array was not created from zarr
        warnings.warn(
            "Could not find from-zarr or load-store-map in dask graph.",
            UserWarning,
            stacklevel=1,
        )
        return None

    # handle only the case when all chunks come from the same zarr array
    unique_key_names = np.unique([k[0] for k in keys])
    if len(unique_key_names) > 1:
        warnings.warn(
            """Dask array was created from multiple zarr arrays,
which is not supported for optimization in multiview-stitcher"
            """,
            UserWarning,
            stacklevel=1,
        )
        return None

    first_value = dask_array.dask[keys[0]]

    # Since 2024.11.2 dask is representing the relevant task
    # below with a Task class instead of a tuple. In 2025.1.0,
    # from-zarr is a Task while load-store-map is (still) a tuple.
    if isinstance(first_value, tuple):
        zarr_array = first_value[1]
    else:
        zarr_array = first_value.args[0].value

    return zarr_array


def get_dask_array_from_slice_into_zarr_array(zarr_array, sl, chunks=None):
    if chunks is not None:
        raise (NotImplementedError)

    d = delayed(lambda x, y: x[y])(zarr_array, sl)
    return da.from_delayed(
        d, shape=tuple([s.stop - s.start for s in sl]), dtype=zarr_array.dtype
    )
