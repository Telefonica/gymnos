#
#
#   Donateacry dataset
#
#

import logging
import os

from ...base import BaseDataset
from ...services.sofia import SOFIA
from .hydra_conf import DonateacryHydraConf
from ...utils.data_utils import extract_archive

from dataclasses import dataclass


@dataclass
class Donateacry(DonateacryHydraConf, BaseDataset):
    """
    TODO: description about data structure

    Parameters
    -----------
    TODO: description of each parameter
    """

    def download(self, root):

        logger = logging.getLogger(__name__)

        download_dir = SOFIA.download_dataset("ferugit/datasets/donateacry", files=["donateacry.zip"])

        logger.info("Extracting donateacry.zip ...")

        extract_archive(os.path.join(download_dir, "donateacry.zip"), root)
        print(root)

