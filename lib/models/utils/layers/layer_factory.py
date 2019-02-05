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
        layerInstance = None
        layerType = config["type"]
        layerSettings = config["settings"]
        if layerType == "convolutional2D": layerInstance = Convolutional2D(layerSettings, self._fw)
        elif layerType == "maxpooling2D": layerInstance = MaxPooling2D(layerSettings, self._fw)
        elif layerType == "flatten": layerInstance = Flatten(layerSettings, self._fw)
        elif layerType == "dense": layerInstance = Dense(layerSettings, self._fw)
        else:
            errMsg = "{0} - Layer suppport for {1} not available.".format(self._log_prefix, layerType)
            self._log.error(errMsg)
            raise ValueError(errMsg)
        
        if layerInstance: 
            return layerInstance