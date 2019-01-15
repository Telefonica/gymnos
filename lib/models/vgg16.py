import logging
import model
import numpy as np
from var.system_paths import *
from var.models import *

class VGG16(model.Model):  
    @property
    def ids(self):
        return self._ids

    def __len__(self):
        return len(self._ids)

    def __init__(self):
        model.Model.__init__(self)
        self._modelId = ID_VGG16
        self._log = logging.getLogger('aitpd')
        self._log_prefix = "VGG16"
        self._modelInstance = None

    def compile(self):

        #TODO: Check all parameters are included, otherwise launch assert or protect code
        if self._modelInstance: 
            self._log.info("{0} - Compiling model...".format(self._log_prefix, self._modelId))
            self._modelInstance.compile( loss=self._compilationOptions["loss"],
                                         optimizer=self._compilationOptions["optimizer"],
                                         metrics=self._compilationOptions["metrics"] )
        else:
            errMsg = "{0} - Model can not be compiled. Model instance does not exist".format(self._log_prefix)
            self._log.error(errMsg)
            raise ValueError(errMsg)                           


    def fit(self, fitSamples, fitLabels, epochs, batch_size, validation_data, verbose):
        self._log.debug("{0} - fitting model ...".format(self._log_prefix))
        self._modelInstance.fit(fitSamples, fitLabels, epochs, batch_size, validation_data, verbose)