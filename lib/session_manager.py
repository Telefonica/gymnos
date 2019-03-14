import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from pydash.objects import get

from .log import logger


class SessionManager(object):

    def __init__(self, sessionConfigFromFile):
        self._log = logger.get_logger()
        self._log_prefix = logger.setup_prefix(__class__)
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
