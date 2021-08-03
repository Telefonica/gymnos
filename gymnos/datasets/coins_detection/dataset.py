#
#
#   Coins Detection dataset
#
#

import os

from ...base import BaseDataset
from ...services.sofia import SOFIA
from ...utils.data_utils import extract_archive
from .hydra_conf import CoinsDetectionHydraConf

from dataclasses import dataclass


@dataclass
class CoinsDetection(CoinsDetectionHydraConf, BaseDataset):
    """
    TODO: description about data structure

    Parameters
    -----------
    TODO: description of each parameter
    """

    def __call__(self, root):
        download_dir = SOFIA.download_dataset("ruben/datasets/coins-detection")
        extract_archive(os.path.join(download_dir, "coins.zip"), root)
