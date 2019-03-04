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

    def __init__(self, run_name=None, experiment_name=None, source_name=None, logdir=None):
        if logdir is not None:
            mlflow.set_tracking_uri(logdir)

        experiment_id = None
        if experiment_name is not None:
            experiment_id = mlflow.create_experiment(experiment_name)

        mlflow.start_run(run_name=run_name, experiment_id=experiment_id, source_name=source_name)


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
