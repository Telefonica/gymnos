#
#
#   Tracker
#
#

import numbers

from keras import callbacks


class Tracker:

    def add_tag(self, tag):
        pass

    def add_tags(self, tags):
        for tag in tags:
            self.add_tag(tag)

    def log_asset(self, name, file_path):
        pass

    def log_image(self, name, file_path):
        pass

    def log_figure(self, name, figure):
        pass

    def log_metric(self, name, value, step=None):
        pass

    def log_metrics(self, dic, prefix=None, step=None):
        prefix = prefix if prefix is not None else ""

        for (name, value) in dic.items():
            self.log_metric(prefix + name, value, step)

    def log_param(self, name, value, step=None):
        pass

    def log_params(self, dic, prefix=None, step=None):
        prefix = prefix if prefix is not None else ""

        for (name, value) in dic.items():
            self.log_param(prefix + name, value, step)

    def log_other(self, name, value):
        pass

    def log_model_graph(self, graph):
        pass

    def get_keras_callback(self, log_params=True, log_metrics=True):
        return KerasCallback(self)  # default callback

    def end(self):
        pass


class KerasCallback(callbacks.Callback):

    def __init__(self, tracker, log_params=True, log_metrics=True):
        self.tracker = tracker
        self.log_params = log_params
        self.log_metrics = log_metrics

        super().__init__()


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

    def add_tag(self, tag):
        for tracker in self.trackers:
            tracker.add_tag(tag)

    def add_tags(self, tags):
        for tracker in self.trackers:
            tracker.add_tags(tags)

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

    def log_other(self, name, value):
        for tracker in self.trackers:
            tracker.log_other(name, value)

    def log_model_graph(self, graph):
        for tracker in self.trackers:
            tracker.log_model_graph(graph)

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
