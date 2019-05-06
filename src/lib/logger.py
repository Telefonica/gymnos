#
#
#   Logger
#
#

import logging


def get_logger(prefix=None):
    """
    Get prefixed logger.

    Parameters
    ----------
    prefix: str, optional
        Prefix to add to logging line

    Returns
    -------
    logger: logging.Logger
        Prefixed logger
    """
    logger = logging.getLogger("gymnosd")

    if prefix is None:
        prefix = ""  # default prefix
    elif not isinstance(prefix, str):
        prefix = prefix.__class__.__name__

    prefixed_logger = logging.LoggerAdapter(logger, extra={"prefix": prefix})
    return prefixed_logger
