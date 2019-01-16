import logging
import model
import numpy as np
from keras.applications.vgg16 import VGG16
from var.system_paths import *
from var.models import *

class VGG16(model.Model):  
    def __init__(self, config):
        model.Model.__init__(self)
        self._log = logging.getLogger('aitpd')
        self._log_prefix = "VGG16"
        self._modelInstance = None
        self._modelId = config["id"]
        self._optimizer = config["options"]["compilation"]["optimizer"]
        self._loss = config["options"]["compilation"]["loss"]
        self._metrics = config["options"]["compilation"]["metrics"]
        self._fineTunning = False
        if "fine-tunning" in config["options"]["custom"]:
            self._fineTunning = True
            self._input_height = config["options"]["custom"]["fine-tunning"]["input_height"]
            self._input_width = config["options"]["custom"]["fine-tunning"]["input_width"]
            self._input_depth = config["options"]["custom"]["fine-tunning"]["input_depth"]
            self._batch_size = config["options"]["custom"]["fine-tunning"]["batch_size"]

    def compile(self):
        #TODO: Check all parameters are included, otherwise launch assert or protect code
        if self._modelInstance: 
            self._log.info("{0} - Compiling model...".format(self._log_prefix, self._modelId))
            self._modelInstance.compile( loss=self._loss,
                                         optimizer=self._optimizer,
                                         metrics=self._metrics )
        else:
            errMsg = "{0} - Model can not be compiled. Model instance does not exist".format(self._log_prefix)
            self._log.error(errMsg)
            raise ValueError(errMsg)                           

    def init(self):
        if self._kerasModel is True: self.__tryToLoadModelFromKeras()
        if self._localModel is True: self.__tryToLoadModelFromLocalVolume()

    def fit(self, fitSamples, fitLabels, epochs, batch_size, validation_data, verbose):
        self._log.debug("{0} - fitting model ...".format(self._log_prefix))
        self._modelInstance.fit(fitSamples, fitLabels, epochs, batch_size, validation_data, verbose)

    def __checkIfSuitableDataSetProperties(self):
        self.__checkNumberOfChannels()
    
    def __tryToLoadModelFromKeras(self):
        if self._fineTunning is True:
            inputShape = (self._input_height, self._input_width, self._input_depth)
            #self._modelInstance = VGG16(include_top=False, input_shape=inputShape)
            self._modelInstance = getattr(self._kerasAppsModule, self._modelNameFromKeras)( include_top=False, input_shape=inputShape)
        else:
            # Model for prediction as: weights="imagenet" and include_top=True by default
            self._modelInstance = getattr(self._kerasAppsModule, self._modelNameFromKeras)()

        '''
         if self._modelNameFromKeras:
            if self._datasetProperties:
                inputShape = ( self._image_height, self._image_width, self.image_depth"] )
                self._modelInstance = getattr(self._kerasAppsModule, self._modelNameFromKeras)( include_top=False,
                                                                                                input_shape=inputShape)
            else:
                self._modelInstance = getattr(self._kerasAppsModule, self._modelNameFromKeras)()
        '''