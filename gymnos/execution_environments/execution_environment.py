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

    @staticmethod
    def add_arguments(parser):
        ...

    @abstractmethod
    def train(self, trainer, **kwargs):
        """
        Train experiment with execution environment
        """
