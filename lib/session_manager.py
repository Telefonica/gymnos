import tensorflow as tf
import logging
from keras.backend.tensorflow_backend import set_session

class SessionManager(object):
    def __init__(self, sessionConfigFromFile):
        self._log = logging.getLogger('aitpd')
        self._log_prefix = "SESSION_MGR"
        self._log.info("{0} - Preparing hardware optimization for training session ...".format(self._log_prefix))
        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = sessionConfigFromFile["num_cores"]
        config.inter_op_parallelism_threads = sessionConfigFromFile["num_physical_cpus"]
        set_session(tf.Session(config=config))      