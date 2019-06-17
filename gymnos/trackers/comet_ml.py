#
#
#   Comet ML Tracker
#
#

from comet_ml import Experiment  # we must import comet.ml before everything
from .tracker import Tracker


class CometML(Tracker):
    """
    Tracker for `Comet.ml <https://www.comet.ml>`_.

    Parameters
    ----------
    api_key: str
        Your API key obtained from comet.ml
    project_name: str, optional
        Send your experiment to a specific project. Otherwise will be sent to *Uncategorized Experiments*.
        If project name does not already exists Comet.ml will create a new project.
    workspace: str, optional
        Attach an experiment to a project that belongs to this workspace.
    """

    def __init__(self, api_key, project_name=None, workspace=None):
        self.api_key = api_key
        self.project_name = project_name
        self.workspace = workspace


    def start(self, run_name, logdir):
        self.experiment = Experiment(api_key=self.api_key, project_name=self.project_name, workspace=self.workspace,
                                     log_code=False, log_graph=False, auto_param_logging=False,
                                     auto_metric_logging=False, parse_args=False, log_env_details=True,
                                     log_git_metadata=False, log_git_patch=False)
        self.experiment.set_name(run_name)

    def log_tag(self, key, value):
        self.experiment.add_tag("{}__{}".format(key, value))


    def log_asset(self, name, file_path):
        self.experiment.log_asset(file_path, file_name=name)


    def log_image(self, name, file_path):
        self.experiment.log_image(file_path, name)


    def log_figure(self, name, figure):
        self.experiment.log_figure(name, figure)


    def log_metric(self, name, value, step=None):
        self.experiment.log_metric(name, value, step)


    def log_param(self, name, value, step=None):
        self.experiment.log_parameter(name, value, step=step)


    def get_keras_callback(self, log_params=True, log_metrics=True):
        return self.experiment.get_keras_callback()


    def end(self):
        self.experiment.end()
