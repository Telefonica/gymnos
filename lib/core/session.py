#
#
#   Session
#
#

import os
import tensorflow as tf

from ..logger import logger


class SessionOptions:

    def __init__(self, allow_memory_growth=False, num_cores=4):
        self.allow_memory_growth = allow_memory_growth
        self.num_cores = num_cores


class Session:

    def __init__(self, device=None, options=None):
        options = options or {}

        self.device = device
        self.options = SessionOptions(**options)

        if self.device == "cpu" and self.has_gpu():
            logger.info("Device has GPU but session is set to CPU")
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        elif self.device == "gpu" and not self.has_gpu():
            logger.warning("Device has not GPU but session is set to GPU")

    def has_gpu(self):
        return bool(tf.test.gpu_device_name())

        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
