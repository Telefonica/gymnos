#
#
#   Pharma Sales dataset
#
#

from ...base import BaseDataset
from .hydra_conf import PharmaSalesHydraConf
from dataclasses import dataclass
from ...utils.data_utils import extract_archive
from gymnos.services import SOFIA
import os



@dataclass
class PharmaSales(PharmaSalesHydraConf, BaseDataset):
    """
    TODO: description about data structure

    Parameters
    -----------
    TODO: description of each parameter
    """

    def download(self, root):
        download_dir =  SOFIA.download_dataset("T4-JARVIS/datasets/Bilbao-Spain")
        extract_archive(os.path.join(download_dir, "Bilbao-Spain.csv"), root)
