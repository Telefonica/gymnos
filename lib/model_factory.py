import logging
from models.vgg16 import *
from var.models import *

class ModelFactory(object):
    def __init__(self, config):
        self._log = logging.getLogger('aitpd')
        self._log_prefix = "MODEL_FACTORY"
        self._modelId = config["id"]
        self._config = config

    def factory(self):
        modelInstance = None
        if self._modelId == ID_VGG16: modelInstance = VGG16(self._config)
        
        if modelInstance is not None:
            self._log.debug("{0} - Instantiating {1} model ...".format(self._log_prefix, self._modelId))
            return modelInstance
        else:
            errMsg = "{0} - Model suppport for {1} not available.".format(self._log_prefix, self._modelId)
            self._log.error(errMsg)
            raise ValueError(errMsg)