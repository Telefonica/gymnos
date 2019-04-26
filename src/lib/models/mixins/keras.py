#
#
#   Keras Mixin
#
#

import os
import keras

from pydoc import locate
from collections.abc import Iterable
from keras.models import load_model

from ...utils.io_utils import read_from_json


KERAS_CALLBACKS_IDS_TO_MODULES_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "var", "keras",
                                                   "callbacks.json")


class KerasMixin:

    def fit(self, X, y, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, shuffle=True,
            class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None):
        """
        Parameters for Keras fit method, more info in `keras <https://keras.io/models/sequential/#fit>`_

        Parameters
        ----------
        X: array_like
            Features
        y: array_like
            Targets.
        epochs: int, optional
            Number of epochs to train.
        batch_size: int, optional
            Number of samples that will be propagated.
        verbose: int, optional
            Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        callbacks: list of dict, optional
            TODO
        validation_split: float, optional
            Fraction of the training data to be used as validation data.
        shuffle: bool, optional
            Whether to shuffle the training data before each epoch
        class_weight: dict, optional
            Dictionnary mapping class indices (integers) to a weight (float) value,
            used for weighting the loss function (during training only).
        sample_weight: array_like, optional
            Array of weights for the training samples, used for weighting the loss function.
        initial_epoch: int, optional
            Epoch at which start training (useful for resuming previous training run).
        steps_per_epoch: int, optional
            Total number of steps (batches of samples) before declaring one epoch finished and starting
            the next epoch.
        validation_steps: int, optional
            Total number of steps (batches of samples) to validate before stopping.
        Returns
        -------
        dict
            Training metrics
        """

        if callbacks is not None:
            callbacks = self.__instantiate_callbacks(callbacks)

        history = self.model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks,
                                 validation_split=validation_split, shuffle=shuffle, class_weight=class_weight,
                                 sample_weight=sample_weight, initial_epoch=initial_epoch,
                                 steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

        return history.history

    def __instantiate_callbacks(self, callbacks_config):
        # directory to save artifacts generated by callbacks
        callbacks_artifacts_dir = os.path.join(os.getcwd(), "callbacks")

        keras_callbacks_ids_to_modules = read_from_json(KERAS_CALLBACKS_IDS_TO_MODULES_PATH)

        callbacks = []
        for callback_config in callbacks_config:
            callback_type = callback_config.pop("type")
            CallbackClass = locate(keras_callbacks_ids_to_modules[callback_type])

            callback_artifacts_path = os.path.join(callbacks_artifacts_dir, callback_type)

            if issubclass(CallbackClass, keras.callbacks.TensorBoard):
                os.makedirs(callback_artifacts_path)
                callback_config["logdir"] = callback_artifacts_path
            elif issubclass(CallbackClass, keras.callbacks.ModelCheckpoint):
                os.makedirs(callback_artifacts_path)
                callback_config["filepath"] = os.path.join(callback_artifacts_path, callback_config["filepath"])
            elif issubclass(CallbackClass, keras.callbacks.CSVLogger):
                os.makedirs(callback_artifacts_path)
                callbacks_config["filename"] = os.path.join(callback_artifacts_path, callback_config["filename"])

            callback = CallbackClass(**callback_config)
            callbacks.append(callback)

        return callbacks


    def evaluate(self, X, y):
        metrics = self.model.evaluate(X, y)
        if not isinstance(metrics, Iterable):
            metrics = [metrics]
        return dict(zip(self.model.metrics_names, metrics))

    def predict(self, X):
        return self.model.predict(X)

    def save(self, directory):
        self.model.save(os.path.join(directory, "model.h5"))

    def restore(self, directory):
        self.model = load_model(os.path.join(directory, "model.h5"))
