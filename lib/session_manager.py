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
        if self._device["type"] is "cpu":
            self._configProto.intra_op_parallelism_threads = self._device["options"]["num_cores"] if "num_cores" in self._device["options"]
            self._configProto.inter_op_parallelism_threads = self._device["options"]["num_physical_cpus"] if "num_physical_cpus" in self._device["options"]
        elif self._device["type"] is "gpu":
            self._configProto["gpu_options"].allow_growth = self._device["options"]["allow_memory_growth"] if "allow_memory_growth" in self._device["options"]

    def __loadSessionSettings(self):
        self._log.debug("{0} - Loading session settings with:\n[\n\t - config protocol = {1}\
                                                    \n]".format( self._log_prefix,
                                                                 self._configProto ) )
        set_session(tf.Session(config=self._configProto))