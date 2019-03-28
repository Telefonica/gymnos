#
#
#   Logger
#
#

import logging


def get_logger(prefix=None):
    logger = logging.getLogger("gymnos")

    if prefix is None:
        prefix = ""  # default prefix
    elif not isinstance(prefix, str):
        prefix = prefix.__class__.__name__

    prefixed_logger = logging.LoggerAdapter(logger, extra={"prefix": prefix})
    return prefixed_logger
