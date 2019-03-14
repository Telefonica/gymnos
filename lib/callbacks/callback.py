from lib.log import logger


class Callback(object):

    def __init__(self):
        self._log = logger.get_logger()
        self._log_prefix = logger.setup_prefix(__class__)
