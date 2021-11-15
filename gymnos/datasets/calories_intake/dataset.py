#
#
#   Calories Intake dataset
#
#

from ...base import BaseDataset
from .hydra_conf import CaloriesIntakeHydraConf
from ...utils.data_utils import extract_archive
from dataclasses import dataclass
from gymnos.services import SOFIA
import os


@dataclass
class CaloriesIntake(CaloriesIntakeHydraConf, BaseDataset):
    """
    TODO: description about data structure

    Parameters
    -----------
    TODO: description of each parameter
    """

    def download(self, root):
        download_dir =  SOFIA.download_dataset("T4-JARVIS/datasets/cal_intake")
        extract_archive(os.path.join(download_dir, "cal_intake.csv"), root)
