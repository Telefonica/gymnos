#
#
#   Tracking
#
#

import os
import logging

from ..trackers import TrackerList

from ..utils.io_utils import import_from_json

TRACKERS_IDS_TO_MODULES_PATH = os.path.join(os.path.dirname(__file__), "..", "var", "trackers.json")

logger = logging.getLogger(__name__)


class Tracking:
    """
    Parameters
    ----------
    log_model_params: bool, optional
        Whether or not log model parameters
    log_model_metrics: bool, optional
        Whether or not log train/test model metrics
    log_training_params: bool, optional
        Whether or not log training params.
    trackers: list of dict, optional
        List of trackers to log parameters and metrics. This property requires a list with dictionnaries with at least
        a ``type`` field specifying the type of tracker. The other properties are the properties for that
        tracker.

        The current available trackers are the following:

            - ``"comet_ml"``: :class:`lib.trackers.comet_ml.CometML`,
            - ``"mlflow"``: :class:`lib.trackers.mlflow.MLFlow`,
            - ``"tensorboard"``: :class:`lib.trackers.tensorboard.Tensorboard`

    additional_params: dict, optional
        Additional parameters to log

    Examples
    --------
    .. code-block:: py

        Tracking(
            log_model_params=True,
            log_model_metrics=True,
            log_training_metrics=False,
            trackers=[
                {
                    "type": "mlflow",
                    "experiment_name": "tfidf_approach"
                },
                {
                    "type": "tensorboard"
                }
            ],
            additional_params={
                data_scientist="Rubén Salas"
            }
        )
    """

    def __init__(self, log_model_params=True, log_model_metrics=True, log_training_params=True, trackers=None,
                 additional_params=None):
        trackers = trackers or []

        self.log_model_params = log_model_params
        self.log_model_metrics = log_model_metrics
        self.log_training_params = log_training_params
        self.additional_params = additional_params or {}

        logger.debug("Import {} trackers".format(len(trackers)))
        self.trackers = TrackerList()
        for tracker_config in trackers:
            TrackerClass = import_from_json(TRACKERS_IDS_TO_MODULES_PATH, tracker_config.pop("type"))
            tracker = TrackerClass(**tracker_config)
            self.trackers.add(tracker)
