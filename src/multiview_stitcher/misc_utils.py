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