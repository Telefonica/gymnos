#
#
#   MLFlow Tracker
#
#

import os
import mlflow
import tempfile

from .tracker import Tracker


class MLFlow(Tracker):

    def __init__(self, experiment_name=None, source_name=None):
        self.source_name = source_name
        self.experiment_id = None
        if experiment_name is not None:
            self.experiment_id = mlflow.create_experiment(experiment_name)


    def start(self, run_name, logdir):
        mlflow.set_tracking_uri(os.path.join(logdir, "mlruns"))
        mlflow.start_run(run_name=run_name, experiment_id=self.experiment_id, source_name=self.source_name)

    def log_metric(self, name, value, step=None):
        mlflow.log_metric(name, value)


    def log_param(self, name, value, step=None):
        mlflow.log_param(name, value)


    def log_other(self, name, value):
        mlflow.set_tag(name, value)


    def log_asset(self, name, file_path):
        mlflow.log_artifact(file_path)


    def log_image(self, name, file_path):
        mlflow.log_artifact(file_path)


    def log_figure(self, name, figure):
        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, name + ".png")
            figure.savefig(path, format="png")
            mlflow.log_artifact(path)


    def end(self):
        mlflow.end_run()
