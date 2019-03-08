import logging

from .models.vgg16 import VGG16
from .models.custom_stack import CustomStack
from .var.models import ID_VGG16, ID_CUSTOM_STACK


class ModelFactory(object):

    def __init__(self, config):
        self._log = logging.getLogger('gymnosd')
        self._log_prefix = "MODEL_FACTORY"
        self._modelId = config["id"]
        self._config = config

    def factory(self):
        modelInstance = None
        if self._modelId == ID_VGG16:
            modelInstance = VGG16(self._config)
        elif self._modelId == ID_CUSTOM_STACK:
            modelInstance = CustomStack(self._config)
        else:
            errMsg = "{0} - Model suppport for {1} not available.".format(self._log_prefix, self._modelId)
            self._log.error(errMsg)
            raise ValueError(errMsg)

        if modelInstance:
            return modelInstance
