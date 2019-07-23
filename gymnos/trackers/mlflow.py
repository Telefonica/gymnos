#
#
#   MLFlow Tracker
#
#

import os
import tempfile

from .tracker import Tracker
from ..utils.lazy_imports import lazy_imports


class MLFlow(Tracker):
    """
    Tracker for `MLflow <https://www.mlflow.org>`_.

    Parameters
    ----------
    experiment_name: str
        Experiment name, must be unique.
    """

    def __init__(self, experiment_name=None):
        self.experiment_id = None
        if experiment_name is not None:
            self.experiment_id = lazy_imports.mlflow.create_experiment(experiment_name)


    def start(self, run_name, logdir):
        lazy_imports.mlflow.set_tracking_uri(os.path.join(logdir, "mlruns"))
        lazy_imports.mlflow.start_run(run_name=run_name, experiment_id=self.experiment_id)

    def log_tag(self, key, value):
        lazy_imports.mlflow.set_tag(key, value)

    def log_metric(self, name, value, step=None):
        lazy_imports.mlflow.log_metric(name, value)


    def log_param(self, name, value, step=None):
        lazy_imports.mlflow.log_param(name, value)


    def log_asset(self, name, file_path):
        lazy_imports.mlflow.log_artifact(file_path)


    def log_image(self, name, file_path):
        lazy_imports.mlflow.log_artifact(file_path)


    def log_figure(self, name, figure):
        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, name + ".png")
            figure.savefig(path, format="png")
            lazy_imports.mlflow.log_artifact(path)


    def end(self):
        lazy_imports.mlflow.end_run()
