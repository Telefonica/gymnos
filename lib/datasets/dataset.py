import os, logging
import numpy as np

class DataSet(object):
    def __init__(self):
        self._log = logging.getLogger('aitpd')
        self._log_prefix = "DATASET"
        self._hdfDataFilename = 'data.hy'
        self._textIdsFilename = 'id.txt'
