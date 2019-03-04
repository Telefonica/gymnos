#
#
#   Comet ML Tracker
#
#

from comet_ml import Experiment  # we must import comet.ml before everything
from .tracker import Tracker


class CometML(Tracker):

    def __init__(self, api_key, project_name=None, workspace=None):
        self.experiment = Experiment(api_key=api_key, project_name=project_name, workspace=workspace,
                                     log_code=False, log_graph=False, auto_param_logging=False,
                                     auto_metric_logging=False, parse_args=False, log_env_details=True,
                                     log_git_metadata=False, log_git_patch=False)


    def add_tag(self, tag):
        self.experiment.add_tag(tag)


    def add_tags(self, tags):
        self.experiment.add_tags(tags)


    def log_asset(self, name, file_path):
        self.experiment.log_asset(file_path, file_name=name)


    def log_image(self, name, file_path):
        self.experiment.log_image(file_path, name)


    def log_figure(self, name, figure):
        self.experiment.log_figure(name, figure)


    def log_metric(self, name, value, step=None):
        self.experiment.log_metric(name, value, step)


    def log_metrics(self, dic, prefix=None, step=None):
        self.experiment.log_metrics(dic, prefix, step)


    def log_param(self, name, value, step=None):
        self.experiment.log_parameter(name, value, step=step)


    def log_params(self, dic, prefix=None, step=None):
        self.experiment.log_parameters(dic, prefix, step)


    def log_other(self, name, value):
        self.experiment.log_other(name, value)


    def log_model_graph(self, graph):
        self.experiment.set_model_graph(graph)


    def get_keras_callback(self, log_params=True, log_metrics=True):
        self.experiment.auto_param_logging = log_params
        self.experiment.auto_metric_logging = log_metrics

        callback = self.experiment.get_keras_callback()

        self.experiment.auto_param_logging = False
        self.experiment.auto_metric_logging = False

        return callback


    def end(self):
        self.experiment.end()
