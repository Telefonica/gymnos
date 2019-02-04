import os,logging

class Callback(object):
    def __init__(self):
        self._log = logging.getLogger('aitpd')
        self._log_prefix = "CALLBACK"