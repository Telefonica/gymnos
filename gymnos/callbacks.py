#
#
#   Callbacks
#
#

import time


class Callback:

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

    def on_train_test_split_begin(self):
        """
        Called when data split into train and test begins.
        """

    def on_train_test_split_end(self):
        """
        Called when data split into train and test ends.
        """

    def on_fit_preprocessors_begin(self):
        """
        Called when fit preprocessors begin.
        """

    def on_fit_preprocessors_end(self):
        """
        Called when fit preprocessors end.
        """

    def on_preprocess_begin(self):
        """
        Called when transform data with preprocessors begin.
        """

    def on_preprocess_end(self):
        """
        Called when transform data with preprocessors end.
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

    def __init__(self, callbacks):
        self.callbacks = callbacks

    def add(self, callback):
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

    def on_train_test_split_begin(self):
        for callback in self.callbacks:
            callback.on_train_test_split_begin()

    def on_train_test_split_end(self):
        for callback in self.callbacks:
            callback.on_train_test_split_end()

    def on_fit_preprocessors_begin(self):
        for callback in self.callbacks:
            callback.on_fit_preprocessors_begin()

    def on_fit_preprocessors_end(self):
        for callback in self.callbacks:
            callback.on_fit_preprocessors_end()

    def on_preprocess_begin(self):
        for callback in self.callbacks:
            callback.on_preprocess_begin()

    def on_preprocess_end(self):
        for callback in self.callbacks:
            callback.on_preprocess_end()

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

    def on_train_begin(self):
        self.times = {}

        self.train_begin_start = time.time()

    def on_train_end(self):
        self.times["train"] = time.time() - self.train_begin_start

    def on_download_and_prepare_data_begin(self):
        self.download_and_prepare_start = time.time()

    def on_download_and_prepare_data_end(self):
        self.times["download_and_prepare_data"] = time.time() - self.download_and_prepare_start

    def on_load_data_begin(self):
        self.load_data_start = time.time()

    def on_load_data_end(self):
        self.times["load_data"] = time.time() - self.load_data_start

    def on_train_test_split_begin(self):
        self.train_test_split_start = time.time()

    def on_train_test_split_end(self):
        self.times["train_test_split"] = time.time() - self.train_test_split_start

    def on_fit_preprocessors_begin(self):
        self.fit_preprocessors_start = time.time()

    def on_fit_preprocessors_end(self):
        self.times["fit_preprocessors"] = time.time() - self.fit_preprocessors_start

    def on_preprocess_begin(self):
        self.transform_preprocessors_start = time.time()

    def on_preprocess_end(self):
        self.times["preprocess"] = time.time() - self.transform_preprocessors_start

    def on_fit_model_begin(self):
        self.fit_model_start = time.time()

    def on_fit_model_end(self):
        self.times["fit_model"] = time.time() - self.fit_model_start

    def on_evaluate_model_begin(self):
        self.evaluate_model_start = time.time()

    def on_evaluate_model_end(self):
        self.times["evaluate_model"] = time.time() - self.evaluate_model_start
