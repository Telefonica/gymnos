#
#
#   Callbacks
#
#

import time
import logging


class Callback:
    """
    Base class for all Gymnos callbacks
    """

    def on_train_begin(self):
        """
        Called when training begins.
        """

    def on_download_and_prepare_data_begin(self):
        """
        Called when dataset is downloading/preparing data
        """

    def on_download_and_prepare_data_end(self):
        """
        Called when dataset finished downloading/preparing data
        """

    def on_load_data_begin(self):
        """
        Called when loading data begins (load data into memory / prepare generator)
        """

    def on_load_data_end(self):
        """
        Called when loading data ends.
        """

    def on_fit_preprocessors_begin(self):
        """
        Called when fit preprocessors begin.
        """

    def on_fit_preprocessors_end(self):
        """
        Called when fit preprocessors ends.
        """

    def on_transform_data_begin(self):
        """
        Called when transform data with preprocessors begins.
        """

    def on_transform_data_end(self):
        """
        Called when transform data with preprocessors ends.
        """

    def on_augment_data_begin(self):
        """
        Called when data augmentation begins
        """

    def on_augment_data_end(self):
        """
        Called when data augmentation ends
        """

    def on_fit_model_begin(self):
        """
        Called when fit model begins.
        """

    def on_fit_model_end(self):
        """
        Called when fit model ends.
        """

    def on_evaluate_model_begin(self):
        """
        Called when evaluate model begins.
        """

    def on_evaluate_model_end(self):
        """
        Called when evaluate model ends.
        """

    def on_train_end(self):
        """
        Called when trainin ends.
        """


class CallbackList:
    """
    Class to execute multiple callbacks at the same time.

    Parameters
    ------------
    callbacks: list of Callback
        Callbacks to add to the queue
    """

    def __init__(self, callbacks):
        self.callbacks = callbacks

    def add(self, callback):
        """
        Add a callback

        Parameters
        -------------
        callback: Callback
            Callback to add to the queue
        """
        self.callbacks.append(callback)

    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_download_and_prepare_data_begin(self):
        for callback in self.callbacks:
            callback.on_download_and_prepare_data_begin()

    def on_download_and_prepare_data_end(self):
        for callback in self.callbacks:
            callback.on_download_and_prepare_data_end()

    def on_load_data_begin(self):
        for callback in self.callbacks:
            callback.on_load_data_begin()

    def on_load_data_end(self):
        for callback in self.callbacks:
            callback.on_load_data_end()

    def on_fit_preprocessors_begin(self):
        for callback in self.callbacks:
            callback.on_fit_preprocessors_begin()

    def on_fit_preprocessors_end(self):
        for callback in self.callbacks:
            callback.on_fit_preprocessors_end()

    def on_transform_data_begin(self):
        for callback in self.callbacks:
            callback.on_transform_data_begin()

    def on_transform_data_end(self):
        for callback in self.callbacks:
            callback.on_transform_data_end()

    def on_augment_data_begin(self):
        for callback in self.callbacks:
            callback.on_augment_data_begin()

    def on_augment_data_end(self):
        for callback in self.callbacks:
            callback.on_augment_data_end()

    def on_fit_model_begin(self):
        for callback in self.callbacks:
            callback.on_fit_model_begin()

    def on_fit_model_end(self):
        for callback in self.callbacks:
            callback.on_fit_model_end()

    def on_evaluate_model_begin(self):
        for callback in self.callbacks:
            callback.on_evaluate_model_begin()

    def on_evaluate_model_end(self):
        for callback in self.callbacks:
            callback.on_evaluate_model_end()

    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end()


class TimeHistory(Callback):
    """
    Class to measure time between steps
    """

    def on_train_begin(self):
        self.times = {}

        self.train_start = time.time()

    def on_train_end(self):
        train_end = time.time()
        self.times["total"] = dict(
            start=self.train_start,
            end=train_end,
            duration=train_end - self.train_start
        )

    def on_download_and_prepare_data_begin(self):
        self.download_and_prepare_start = time.time()

    def on_download_and_prepare_data_end(self):
        download_and_prepare_end = time.time()

        self.times["download_and_prepare_data"] = dict(
            start=self.download_and_prepare_start,
            end=download_and_prepare_end,
            duration=download_and_prepare_end - self.download_and_prepare_start
        )

    def on_load_data_begin(self):
        self.load_data_start = time.time()

    def on_load_data_end(self):
        download_and_prepare_end = time.time()
        self.times["load_data"] = dict(
            start=self.load_data_start,
            end=download_and_prepare_end,
            duration=download_and_prepare_end - self.load_data_start
        )

    def on_fit_preprocessors_begin(self):
        self.fit_preprocessors_start = time.time()

    def on_fit_preprocessors_end(self):
        fit_preprocessors_end = time.time()
        self.times["fit_preprocessors"] = dict(
            start=self.fit_preprocessors_start,
            end=fit_preprocessors_end,
            duration=fit_preprocessors_end - self.fit_preprocessors_start
        )

    def on_transform_data_begin(self):
        self.transform_preprocessors_start = time.time()

    def on_transform_data_end(self):
        transform_preprocessors_end = time.time()
        self.times["transform_data"] = dict(
            start=self.transform_preprocessors_start,
            end=transform_preprocessors_end,
            duration=transform_preprocessors_end - self.transform_preprocessors_start
        )

    def on_augment_data_begin(self):
        self.data_augmentation_start = time.time()

    def on_augment_data_end(self):
        data_augmentation_end = time.time()
        self.times["augment_data"] = dict(
            start=self.data_augmentation_start,
            end=data_augmentation_end,
            duration=data_augmentation_end - self.data_augmentation_start
        )

    def on_fit_model_begin(self):
        self.fit_model_start = time.time()

    def on_fit_model_end(self):
        fit_model_end = time.time()
        self.times["fit_model"] = dict(
            start=self.fit_model_start,
            end=fit_model_end,
            duration=fit_model_end - self.fit_model_start
        )

    def on_evaluate_model_begin(self):
        self.evaluate_model_start = time.time()

    def on_evaluate_model_end(self):
        evaluate_model_end = time.time()
        self.times["evaluate_model"] = dict(
            start=self.evaluate_model_start,
            end=evaluate_model_end,
            duration=evaluate_model_end - self.evaluate_model_start
        )


class Logger(Callback):

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def on_train_begin(self):
        self.logger.info("Training has begun")

    def on_download_and_prepare_data_begin(self):
        self.logger.info("Downloading and preparing data has begun")

    def on_download_and_prepare_data_end(self):
        self.logger.info("Downloading and preparing data has ended")

    def on_load_data_begin(self):
        self.logger.info("Data loading has begun")

    def on_load_data_end(self):
        self.logger.info("Data loading has ended")

    def on_fit_preprocessors_begin(self):
        self.logger.info("Preprocessors fitting has begun")

    def on_fit_preprocessors_end(self):
        self.logger.info("Preprocessors fitting has ended")

    def on_transform_data_begin(self):
        self.logger.info("Preprocessing has begun")

    def on_transform_data_end(self):
        self.logger.info("Preprocessing has ended")

    def on_augment_data_begin(self):
        self.logger.info("Data augmentation has begun")

    def on_augment_data_end(self):
        self.logger.info("Data augmentation has ended")

    def on_fit_model_begin(self):
        self.logger.info("Model fitting has begun")

    def on_fit_model_end(self):
        self.logger.info("Model fitting has ended")

    def on_evaluate_model_begin(self):
        self.logger.info("Model evaluation has begun")

    def on_evaluate_model_end(self):
        self.logger.info("Model evaluation has ended")

    def on_train_end(self):
        self.logger.info("Training has ended")
