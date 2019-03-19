#
#
#   Training
#
#

import os

from pydoc import locate

from ..logger import logger
from ..utils.io_utils import read_from_json

CALLBACKS_IDS_TO_MODULES_PATH = os.path.join(os.path.dirname(__file__), "..", "var", "callbacks.json")


class TrainingSamples:

    def __init__(self, fit, test=0, val=0):
        if fit + test + val > 1.0:
            raise ValueError("Samples of Fit + test + val must be lower than 1.0")
        if fit + test + val < 1.0:
            logger.warning("Fit + test + val is lower than 1.0")

        self.fit = fit
        self.val = val
        self.test = test



class Training:

    def __init__(self, samples, batch_size=32, epochs=10, callbacks=None):
        callbacks = callbacks or []

        self.epochs = epochs
        self.batch_size = batch_size
        self.samples = TrainingSamples(**samples)

        self.callbacks = []
        for callback_config in callbacks:
            CallbackClass = self.__retrieve_callback_from_type(callback_config.pop("type"))
            callback = CallbackClass(**callback_config)
            self.callbacks.append(callback)

    def __retrieve_callback_from_type(self, callback_id):
        callbacks_ids_to_modules = read_from_json(CALLBACKS_IDS_TO_MODULES_PATH)
        callback_loc = callbacks_ids_to_modules[callback_id]
        return locate(callback_loc)
