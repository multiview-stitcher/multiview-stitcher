import logging
from contextlib import contextmanager


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
