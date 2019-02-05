import os

from datasets.kaggle_dogs_vs_cats import *
from var.datasets import *

class KaggleFactory(object):
    def __init__(self, config):
        self._log = logging.getLogger('gymnosd')
        self._log_prefix = "KAGGLE_FACTORY"
        self._config = config
        self._dataSetId = config["properties"]["kaggle_source"]["id"]

    def factory(self):
        dsInstance = None
        if self._dataSetId == KAGGLE_DOGS_VS_CATS: dsInstance = DogsVsCats(self._config)
        else:
            errMsg = "{0} - Kaggle data set suppport for {1} not available.".format(self._log_prefix, self._dataSetId)
            self._log.error(errMsg)
            raise ValueError(errMsg)

        if dsInstance:
            return dsInstance