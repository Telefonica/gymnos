#
#
#   Training
#
#

import os

from pydoc import locate
from keras import callbacks

from ..logger import get_logger
from ..utils.io_utils import read_from_json

CALLBACKS_IDS_TO_MODULES_PATH = os.path.join(os.path.dirname(__file__), "..", "var", "callbacks.json")


class TrainingSamples:

    def __init__(self, fit, test=0, val=0):
        self.logger = get_logger(prefix=self)
        if fit + test + val > 1.0:
            raise ValueError("Samples of Fit + test + val must be lower than 1.0")
        if fit + test + val < 1.0:
            self.logger.warning("Samples of Fit + test + val is lower than 1.0")

        if fit <= 0.0:
            raise ValueError("Fit samples must be greater than 0")

        self.fit = fit
        self.val = val
        self.test = test


class Training:

    def __init__(self, samples, batch_size=32, epochs=10, callbacks=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.samples = TrainingSamples(**samples)

        self.callbacks = []
        self.callbacks_config = callbacks or []


    def configure_callbacks(self, base_dir):
        for callback_config in self.callbacks_config:
            callback_type = callback_config.pop("type")
            CallbackClass = self.__retrieve_callback_from_type(callback_type)

            callback_dir = os.path.join(base_dir, callback_type)

            if issubclass(CallbackClass, callbacks.TensorBoard):
                os.makedirs(callback_dir, exist_ok=True)
                callback_config["log_dir"] = os.path.join(callback_dir, callback_config["log_dir"])
            elif issubclass(CallbackClass, callbacks.ModelCheckpoint):
                os.makedirs(callback_dir, exist_ok=True)
                callback_config["filepath"] = os.path.join(callback_dir, callback_config["filepath"])
            elif issubclass(CallbackClass, callbacks.CSVLogger):
                os.makedirs(callback_dir, exist_ok=True)
                callback_config["filename"] = os.path.join(callback_dir, callback_config["filename"])

            callback = CallbackClass(**callback_config)
            self.callbacks.append(callback)

    def __retrieve_callback_from_type(self, callback_id):
        callbacks_ids_to_modules = read_from_json(CALLBACKS_IDS_TO_MODULES_PATH)
        callback_loc = callbacks_ids_to_modules[callback_id]
        return locate(callback_loc)
