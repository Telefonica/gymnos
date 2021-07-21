#
#
#   Reuters dataset
#
#

from ...base import BaseDataset
from .hydra_conf import ReutersHydraConf

from dataclasses import dataclass


@dataclass
class Reuters(ReutersHydraConf, BaseDataset):
    """
    TODO: description about data structure

    Parameters
    -----------
    TODO: description of each parameter
    """

    def __call__(self, root):
        pass  # TODO: save dataset files to `root`
