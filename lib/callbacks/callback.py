import logging


class Callback(object):

    def __init__(self):
        self._log = logging.getLogger('gymnosd')
        self._log_prefix = "CALLBACK"
