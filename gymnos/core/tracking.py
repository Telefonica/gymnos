#
#
#   Tracking
#
#

import uuid
import logging

from copy import deepcopy

from ..trackers.tracker import TrackerList
from ..utils.py_utils import cached_property

logger = logging.getLogger(__name__)


class Tracking:
    """
    Parameters
    ----------
    run_id: str, optional
        ID of the run to log under. If not provided, it will be a random UUID
    tags: dict, optional
        Tags for current run. The keys will be the tag names and the values will be the tag values.
    log_params: bool, optional
        Whether or not log model parameters
    log_metrics: bool, optional
        Whether or not log train/test model metrics
    log_training_params: bool, optional
        Whether or not log training params.
    trackers: list of dict, optional
        List of trackers to log parameters and metrics. This property requires a list with dictionnaries with at least
        a ``type`` field specifying the type of tracker. The other properties are the arguments for the constructor of that tracker.

    Examples
    --------
    .. code-block:: py

        Tracking(
            tags={
                "user": "John Doe"
            },
            log_params=True,
            log_metrics=True,
            log_training_metrics=False,
            trackers=[
                {
                    "type": "mlflow",
                    "experiment_name": "tfidf_approach"
                },
                {
                    "type": "tensorboard"
                }
            ]
        )
    """  # noqa: E501

    def __init__(self, run_id=None, tags=None, log_params=True, log_metrics=True, log_training_params=True,
                 trackers=None):
        trackers = trackers or []

        if run_id is None:
            run_id = uuid.uuid4().hex

        self.tags = tags or {}
        self.run_id = run_id
        self.log_params = log_params
        self.log_metrics = log_metrics
        self.log_training_params = log_training_params

        self._trackers_spec = deepcopy(trackers)

    @cached_property
    def trackers(self):
        return TrackerList.from_dict(self._trackers_spec)

    def to_dict(self):
        return dict(
            run_id=self.run_id,
            tags=self.tags,
            log_params=self.log_params,
            log_metrics=self.log_metrics,
            log_training_params=self.log_training_params,
            trackers=self._trackers_spec
        )
