import tensorflow as tf
import logging

from pydash.objects import get
from keras.backend.tensorflow_backend import set_session


class SessionManager(object):

    def __init__(self, sessionConfigFromFile):
        self._log = logging.getLogger('gymnosd')
        self._log_prefix = "SESSION_MGR"
        self._log.info("{0} - Setting up device options for training session ...".format(self._log_prefix))
        self._configProto = tf.ConfigProto()
        self._device = sessionConfigFromFile["device"]
        self.__parseDeviceOptions()
        self.__loadSessionSettings()

    def __parseDeviceOptions(self):
        if self._device["type"] == "cpu":
            self._configProto.intra_op_parallelism_threads = get(self._device, "options.num_cores")
            self._configProto.intra_op_parallelism_threads = get(self._device, "options.num_physical_cpus")
        elif self._device["type"] == "gpu":
            self._configProto.gpu_options.allow_growth = get(self._device, "options.allow_memory_growth")

    def __loadSessionSettings(self):
        aux = ""
        for attr in self._device["options"]:
            aux += "\n\t - {0} = {1}".format(attr, self._device["options"][attr])
        self._log.debug("{0} - Loading session settings for [{1}] device with:\n[{2}\n]".format(
                        self._log_prefix, self._device["type"], aux))
        set_session(tf.Session(config=self._configProto))
