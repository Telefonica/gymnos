from .datasets.kaggle_dogs_vs_cats import KaggleDogsVsCats
from .datasets.mnist import MNIST
from .log import logger
from .var.datasets import MNIST_DIGITS, KAGGLE_DOGS_VS_CATS


class DataSetFactory(object):
    def __init__(self, config):
        self._log = logger.get_logger()
        self._log_prefix = logger.setup_prefix(__class__)
        self._config = config
        self._dataSetId = config["id"]

    def factory(self):
        dsInstance = None
        if self._dataSetId == MNIST_DIGITS:
            dsInstance = MNIST(self._config)
        elif self._dataSetId  == KAGGLE_DOGS_VS_CATS:
            dsInstance = KaggleDogsVsCats(self._config)
        else:
            errMsg = "{0} - Data set suppport for {1} not available.".format(self._log_prefix, self._dataSetId)
            self._log.error(errMsg)
            raise ValueError(errMsg)

        if dsInstance:
            return dsInstance
