import logging, inspect, os, importlib
import keras

from var.system_paths import *

class Model(object):
    def __init__(self):
        self._log = logging.getLogger('aitpd')
        self._log_prefix = "MODEL"
        self._kerasAppsModule = importlib.import_module("keras.applications")
        self._modelNameFromKeras = None
        self._datasetProperties = None
        self._compilationOptions = None
        self._kerasModel = False
        self._localModel = False

    def addCompilationOptions(self, compilationOptions):
        self._compilationOptions = compilationOptions

    def addDatasetProperties(self, dsProperties):
        self._datasetProperties = dsProperties

    def init(self):
        if self._kerasModel is True: self.__tryToLoadModelFromKeras()
        if self._localModel is True: self.__tryToLoadModelFromLocalVolume()

    def lookForModelSource(self):
        if self.__modelInKeras():
            self._log.info("{0} - Model '{1}' found in keras.".format(self._log_prefix, self._modelId))
            self._kerasModel = True
        else:
            self._log.warning("{0} - Model '{1}' not found in keras.".format(self._log_prefix, self._modelId))
            if self.__modelInLocalVolume():
                self._log.info("{0} - Model '{1}' found in local volume.".format(self._log_prefix, self._modelId))
                self._localModel = True
            else:
                self._log.warning("{0} - Model '{1}' not found in local volume.".format(self._log_prefix, self._modelId))
                self.__loadModelFromConfig()

    def lookForPretrainedWeights(self):
        self._log.debug("{0} - lookForPretrainedWeights - looking for pretrained weights".format(self._log_prefix))
        # lookup and load

    def summary(self):
        if self._modelInstance: 
            self._log.info("{0} - Model summary.".format(self._log_prefix))
            self._modelInstance.summary()
        else:
            errMsg = "{0} - Can not display model summary. Model instance does not exist".format(self._log_prefix)
            self._log.error(errMsg)
            raise ValueError(errMsg)                           


    def __tryToLoadModelFromKeras(self):
        if self._modelNameFromKeras:
            if self._datasetProperties:
                inputShape = ( self._datasetProperties["image_height"], 
                               self._datasetProperties["image_width"],
                               self._datasetProperties["image_depth"] )
                inputShape = (24,24,1)
                self._modelInstance = getattr(self._kerasAppsModule, self._modelNameFromKeras)( include_top=False,
                                                                                                input_shape=inputShape)
            else:
                self._modelInstance = getattr(self._kerasAppsModule, self._modelNameFromKeras)()

    def __modelInKeras(self):
        retval = False
        for name, data in inspect.getmembers(self._kerasAppsModule):
            if name == '__builtins__':
                continue
            if name == self._modelId:
                retval = True
                self._log.debug("{0} - Model - __modelInKeras - name: {1} - {2}".format(self._log_prefix, name, repr(data)))
                members = inspect.getmembers(data)
                self._modelNameFromKeras = members[0][0]      
                
        return retval

    def __modelInLocalVolume(self):
        retval = False
        targetDir = '{0}/{1}'.format(MODELS_PATH, self._modelId)
        if os.path.isdir(targetDir):
            retval = True
        return retval

    def __tryToLoadModelFromLocalVolume(self):
        pass

    def __loadModelFromConfig(self):
        pass