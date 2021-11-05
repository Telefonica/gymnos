#
#
#   Hello World dataset
#
#

from ...base import BaseDataset
from .hydra_conf import HelloWorldHydraConf

from dataclasses import dataclass


@dataclass
class HelloWorld(HelloWorldHydraConf, BaseDataset):
    """
    TODO: description about data structure

    Parameters
    -----------
    TODO: description of each parameter
    """

    def download(self, root):
        pass  # TODO: save dataset files to `root`
