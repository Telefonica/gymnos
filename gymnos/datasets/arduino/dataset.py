#
#
#   Arduino dataset
#
#

from ...base import BaseDataset
from .hydra_conf import ArduinoHydraConf

from dataclasses import dataclass


@dataclass
class Arduino(ArduinoHydraConf, BaseDataset):
    """
    TODO: description about data structure

    Parameters
    -----------
    TODO: description of each parameter
    """

    def download(self, root):
        download_dir =  SOFIA.download_dataset("IndIAna_jones/datasets/Arduino_environmental_data")
        pass  # TODO: save dataset files to `root`
