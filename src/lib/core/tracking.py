#
#
#   Tracking
#
#

import os

from pydoc import locate

from ..trackers import TrackerList, Tensorboard, MLFlow

from ..utils.io_utils import read_from_json

TRACKERS_IDS_TO_MODULES_PATH = os.path.join(os.path.dirname(__file__), "..", "var", "trackers.json")


class Tracking:

    def __init__(self, log_model_params=True, log_model_metrics=True, trackers=None, params=None):
        trackers = trackers or []

        self.log_model_params = log_model_params
        self.log_model_metrics = log_model_metrics
        self.params = params or {}

        self.trackers = TrackerList()
        self.trackers_config = trackers


    def configure_trackers(self, logdir, run_name):
        for tracker_config in self.trackers_config:
            tracker_type = tracker_config.pop("type")

            TrackerClass = self.__retrieve_tracker_from_type(tracker_type)
            if issubclass(TrackerClass, Tensorboard):
                tracker_config["logdir"] = os.path.join(logdir, "tensorboard", run_name)
            elif issubclass(TrackerClass, MLFlow):
                tracker_config["run_name"] = run_name
                tracker_config["logdir"] = os.path.join(logdir, "mlruns")

            tracker = TrackerClass(**tracker_config)
            self.trackers.add(tracker)


    def __retrieve_tracker_from_type(self, tracker_type):
        trackers_ids_to_modules = read_from_json(TRACKERS_IDS_TO_MODULES_PATH)
        tracker_loc = trackers_ids_to_modules[tracker_type]
        return locate(tracker_loc)


    def get_keras_callbacks(self):
        callbacks = self.trackers.get_keras_callbacks(log_params=self.log_model_params,
                                                      log_metrics=self.log_model_metrics)
        return callbacks
