#
#
#   Tiny Sounds dataset
#
#

from ...base import BaseDataset
from .hydra_conf import TinySoundsHydraConf
from ...utils.data_utils import extract_archive
from gymnos.services import SOFIA
from dataclasses import dataclass
import logging
import os

@dataclass
class TinySounds(TinySoundsHydraConf, BaseDataset):
    """
    The dataset is a zip of wavs already splitted and augmented with noise
    """

    def download(self, root):
        logger = logging.getLogger(__name__)
        logger.info("Donwloading Sounds ...")
        download_dir = SOFIA.download_dataset(
            "IndIAna_jones/datasets/tiny_sounds")

        logger.info("Extracting Sounds ...")
        extract_archive(os.path.join(download_dir, "audios.zip"), root)
    
