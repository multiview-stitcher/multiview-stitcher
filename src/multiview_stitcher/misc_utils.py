import inspect
import logging
from contextlib import contextmanager

import numpy as np


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


from itertools import islice


def requires_overlap(overlap_fn):
    """
    Decorator that attaches a required-overlap calculator to a fusion or
    weights function.

    ``overlap_fn`` receives a dict of the function's fully-resolved keyword
    arguments (defaults merged with whatever the caller passes) and must
    return the required overlap in pixels as an int.

    Example
    -------
    >>> @requires_overlap(lambda kwargs: 2 * kwargs["sigma_2"])
    ... def my_weights_func(transformed_views, blending_weights, sigma_2=11):
    ...     ...

    The decorator preserves the original function's signature and name, and
    attaches a ``required_overlap(kwargs)`` attribute that can be called
    from ``fuse()`` via::

        if hasattr(func, "required_overlap"):
            overlap = func.required_overlap(func_kwargs or {})
    """
    def decorator(func):
        sig = inspect.signature(func)

        def required_overlap(kwargs):
            defaults = {
                k: v.default
                for k, v in sig.parameters.items()
                if v.default is not inspect.Parameter.empty
            }
            return overlap_fn({**defaults, **(kwargs or {})})

        func.required_overlap = required_overlap
        return func

    return decorator


def requires_source_shrinkage(shrinkage_fn):
    """
    Decorator that attaches a source-shrinkage calculator to a fusion function.

    ``shrinkage_fn`` receives a dict of the function's fully-resolved keyword
    arguments (defaults merged with whatever the caller passes) and must
    return the required source shrinkage as a float (isotropic, in physical
    units) or as a dict mapping dimension names to floats (per-dimension).

    The shrinkage causes blending weights to reach zero that many physical
    units *before* the input view borders, preventing border artefacts from
    convolution-based operations (e.g. multi-view deconvolution with a PSF).

    Example
    -------
    >>> @requires_source_shrinkage(lambda kwargs: kwargs["border_exclusion"])
    ... def my_fusion_func(transformed_views, blending_weights,
    ...                    border_exclusion=5.0):
    ...     ...

    The decorator preserves the original function's signature and name, and
    attaches a ``required_source_shrinkage(kwargs)`` attribute that can be
    called from ``fuse()`` via::

        if hasattr(func, "required_source_shrinkage"):
            shrink_distance = func.required_source_shrinkage(func_kwargs or {})
    """
    def decorator(func):
        sig = inspect.signature(func)

        def required_source_shrinkage(kwargs):
            defaults = {
                k: v.default
                for k, v in sig.parameters.items()
                if v.default is not inspect.Parameter.empty
            }
            return shrinkage_fn({**defaults, **(kwargs or {})})

        func.required_source_shrinkage = required_source_shrinkage
        return func

    return decorator


def ndindex_batches(nblocks, batch_size):
    it = np.ndindex(*nblocks)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        yield batch


def process_batch_using_ray(func, block_ids, num_cpus=4):
    """
    Process a batch of block_ids using ray for parallelization.
    """

    try:
        import ray
    except ImportError:
        raise ImportError("Please install ray to use this function.")

    if not ray.is_initialized():
        ray.init(
            include_dashboard=True,  # make sure the dashboard starts
            # dashboard_port=8265,      # optional: specify port
            num_cpus=num_cpus
        )

    futures = [ray.remote(func).remote(block_id) for block_id in block_ids]
    ray.get(futures)

    return


def process_batch_using_joblib(
    func,
    block_ids,
    n_jobs=4,
    backend='loky',
    ):
    """
    A batch function that uses joblib for parallel processing.
    1. func: function to apply to each block_id
    2. block_ids: list of block IDs to process
    3. n_jobs: number of parallel jobs to run
    4. backend: joblib backend to use ('threading' or 'loky' (default) for multiprocessing)
    """

    try:
        from joblib import Parallel, delayed
    except ImportError:
        raise ImportError("Please install joblib to use this function.")

    Parallel(
        n_jobs=n_jobs,
        backend=backend
        )(
        delayed(func)(block_id) for block_id in block_ids
    )
    return
