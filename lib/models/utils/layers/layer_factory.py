import logging
from convolutional2D import *
from maxpooling2D import *
from flatten import *
from dense import *

class LayerFactory(object):
    def __init__(self, framework):
        self._log = logging.getLogger('gymnosd')
        self._log_prefix = "LAYER_FACTORY"
        self._fw = framework

    def factory(self, layerConfig):
        layerWrapper = None
        layerType = layerConfig["type"]
        layerSettings = layerConfig["settings"]
        if layerType == "convolutional2D": layerWrapper = Convolutional2D(layerSettings, self._fw)
        elif layerType == "maxpooling2D": layerWrapper = MaxPooling2D(layerSettings, self._fw)
        elif layerType == "flatten": layerWrapper = Flatten(layerSettings, self._fw)
        elif layerType == "dense": layerWrapper = Dense(layerSettings, self._fw)
        else:
            errMsg = "{0} - Layer suppport for {1} not available.".format(self._log_prefix, layerType)
            self._log.error(errMsg)
            raise ValueError(errMsg)
        
        if layerWrapper: 
            return layerWrapper.getInstance()