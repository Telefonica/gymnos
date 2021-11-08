#
#
#   Hello Dataset dataset
#
#

from ...base import BaseDataset
from .hydra_conf import HelloDatasetHydraConf

from dataclasses import dataclass


@dataclass
class HelloDataset(HelloDatasetHydraConf, BaseDataset):
    """
    TODO: description about data structure

    Parameters
    -----------
    TODO: description of each parameter
    """

    def download(self, root):
        pass  # TODO: save dataset files to `root`
