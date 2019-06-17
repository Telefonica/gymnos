#
#
#   Tracker
#
#

import numbers

from keras import callbacks
from collections.abc import Iterable
from abc import ABCMeta, abstractmethod


class Tracker(metaclass=ABCMeta):

    @abstractmethod
    def start(self, run_id, logdir):
        """
        Called when the experiment is started.

        Note
        -----
        Useful for initialization purposes

        Parameters
        ----------
        run_id: str
            ID identifying the experiment's run.
        logdir: str
            Path of logging.
        """

    @abstractmethod
    def log_tag(self, key, value):
        """
        Log tag to experiment.

        Parameters
        ----------
        key: str
            Tag name.
        value: str
            Tag value.
        """

    def log_tags(self, tags):
        for key, value in tags.items():
            self.log_tag(key, value)

    @abstractmethod
    def log_asset(self, name, file_path):
        """
        Log asset.

        Parameters
        ----------
        name: str
            Asset's name
        file_path: str
            Asset's path.
        """

    @abstractmethod
    def log_image(self, name, file_path):
        """
        Log image

        Parameters
        ----------
        name: str
            Image's name.
        file_path: str
            Image's path.
        """

    @abstractmethod
    def log_figure(self, name, figure):
        """
        Log Matplotlib figure

        Parameters
        ----------
        name: str
            Figure's name.
        figure: matplotlib.Figure
            Matplotlib figure.
        """

    @abstractmethod
    def log_metric(self, name, value, step=None):
        """
        Log metric

        Parameters
        ----------
        name: str
            Metric's name.
        value: any
            Metric's value.
        step: int
            Metric's step.
        """

    def __log_metric_list(self, name, metric_list):
        for step, val in enumerate(metric_list):
            self.log_metric(name, val, step)

    def log_metrics(self, dic, prefix=None, step=None):
        prefix = prefix if prefix is not None else ""

        for (name, value) in dic.items():
            if isinstance(value, Iterable):
                self.__log_metric_list(prefix + name, value)
            else:
                self.log_metric(prefix + name, value, step)

    @abstractmethod
    def log_param(self, name, value, step=None):
        """
        Log parameter

        Parameters
        ----------
        name: str
            Parameter's name.
        value: any
            Parameter's value.
        """

    def log_params(self, dic, prefix=None, step=None):
        prefix = prefix if prefix is not None else ""

        for (name, value) in dic.items():
            self.log_param(prefix + name, value, step)


    def get_keras_callback(self, log_params=True, log_metrics=True):
        return KerasCallback(self)  # default callback

    @abstractmethod
    def end(self):
        """
        Called when the experiment is finished.

        Note
        -----
        Useful to release/write artifacts.
        """


class KerasCallback(callbacks.Callback):

    def __init__(self, tracker, log_params=True, log_metrics=True):
        self.tracker = tracker
        self.log_params = log_params
        self.log_metrics = log_metrics


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.log_metrics:
            metrics = {k: v for k, v in logs.items() if isinstance(v, numbers.Number)}
            self.tracker.log_metrics(metrics, step=epoch)

    def on_train_begin(self, logs=None):
        if not self.log_params:
            return

        self.tracker.log_params(logs or {})

        params_to_ignore = ["verbose", "do_validation", "validation_steps"]

        if hasattr(self, "params") and self.params:
            params = {k: v for k, v in self.params.items() if k != "metrics" and k not in params_to_ignore}
            self.tracker.log_params(params)


class TrackerList:

    def __init__(self, trackers=None):
        self.trackers = trackers or []

    def add(self, tracker):
        self.trackers.append(tracker)

    def reset(self):
        self.trackers = []

    def start(self, run_id, logdir):
        for tracker in self.trackers:
            tracker.start(run_id, logdir)

    def log_tag(self, key, value):
        for tracker in self.trackers:
            tracker.log_tag(key, value)

    def log_tags(self, tags):
        for tracker in self.trackers:
            tracker.log_tags(tags)

    def log_asset(self, name, file_path):
        for tracker in self.trackers:
            tracker.log_asset(name, file_path)

    def log_image(self, name, file_path):
        for tracker in self.trackers:
            tracker.log_asset(name, file_path)

    def log_figure(self, name, figure):
        for tracker in self.trackers:
            tracker.log_figure(name, figure)

    def log_metric(self, name, value, step=None):
        for tracker in self.trackers:
            tracker.log_metric(name, value, step)

    def log_metrics(self, dic, prefix=None, step=None):
        for tracker in self.trackers:
            tracker.log_metrics(dic, prefix, step)

    def log_param(self, name, value, step=None):
        for tracker in self.trackers:
            tracker.log_param(name, value, step)

    def log_params(self, dic, prefix=None, step=None):
        for tracker in self.trackers:
            tracker.log_params(dic, prefix, step)

    def get_keras_callbacks(self, log_params=True, log_metrics=True):
        callbacks = []
        for tracker in self.trackers:
            callback = tracker.get_keras_callback(log_params, log_metrics)
            callbacks.append(callback)
        return callbacks

    def end(self):
        for tracker in self.trackers:
            tracker.end()

    def __len__(self):
        return len(self.trackers)
