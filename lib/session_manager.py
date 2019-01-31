import tensorflow as tf
import logging
from keras.backend.tensorflow_backend import set_session

class SessionManager(object):
    def __init__(self, sessionConfigFromFile):
        self._log = logging.getLogger('aitpd')
        self._log_prefix = "SESSION_MGR"
        self._log.info("{0} - Setting up device options for training session ...".format(self._log_prefix))
        self._configProto = tf.ConfigProto()
        self._device = sessionConfigFromFile["device"]
        self.__parseDeviceOptions()
        self.__loadSessionSettings()
        
    def __parseDeviceOptions(self):
        if self._device["type"] == "cpu":
            self._configProto.intra_op_parallelism_threads = self._device["options"]["num_cores"] if "num_cores" in self._device["options"] else None
            self._configProto.inter_op_parallelism_threads = self._device["options"]["num_physical_cpus"] if "num_physical_cpus" in self._device["options"] else None
        elif self._device["type"] == "gpu":
            self._configProto.gpu_options.allow_growth = self._device["options"]["allow_memory_growth"] if "allow_memory_growth" in self._device["options"] else None

    def __loadSessionSettings(self):
        aux = ""
        for attr in self._device["options"]:  
            aux+="\n\t - {0} = {1}".format(attr, self._device["options"][attr])
        self._log.debug("{0} - Loading session settings for [{1}] device with:\n[{2}\n]".format( self._log_prefix, self._device["type"], aux))
        set_session(tf.Session(config=self._configProto))
        