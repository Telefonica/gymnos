#
#
#   Execution environment
#
#

from .. import config

from abc import ABCMeta, abstractmethod


class ExecutionEnvironment(metaclass=ABCMeta):
    """
    Abstract class for execution environmnents

    Parameters
    ------------
    trainer: gymnos.trainer.Trainer
        Trainer instance
    config_files: list of str, optional
        List of JSON paths to look for configuration variables
    """

    class Config(config.Config):
        """
        Configuration variables for execution environment
        """

    def __init__(self, config_files=None):
        self.config = self.Config(files=config_files)
        self.config.load()

    @abstractmethod
    def train(self, trainer):
        """
        Train experiment with execution environment
        """

    def monitor(self, **train_kwargs):
        """
        Monitor training status

        Parameters
        -----------
        train_kwargs: any
            train() method outputs.
        """
