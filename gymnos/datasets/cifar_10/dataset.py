#
#
#   Cifar 10 dataset
#
#
import logging
import os
import glob

from ...base import BaseDataset
from ...services.sofia import SOFIA
from .hydra_conf import CIFAR10HydraConf
from ...utils.data_utils import extract_archive

from dataclasses import dataclass


@dataclass
class CIFAR10(CIFAR10HydraConf, BaseDataset):
    """
    TODO: description about data structure

    Parameters
    -----------
    TODO: description of each parameter
    """

    def __call__(self, root):
        logger = logging.getLogger(__name__)

        logger.info("Downloading dataset ...")
        download_dir = SOFIA.download_dataset("ruben/datasets/CIFAR-10")

        for fpath in glob.iglob(os.path.join(download_dir, "*.zip")):
            logger.info(f"Extracting {os.path.basename(fpath)} ...")
            extract_archive(fpath, root)
