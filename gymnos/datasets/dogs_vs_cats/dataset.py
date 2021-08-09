#
#
#   Dataset
#
#

import os
import logging

from ...base import BaseDataset
from .hydra_conf import DogsVsCatsHydraConf
from ...services.sofia import SOFIA
from ...utils.data_utils import extract_archive

from dataclasses import dataclass


@dataclass
class DogsVsCats(DogsVsCatsHydraConf, BaseDataset):
    """
    Data has the following structure:

    .. code-block::

        dog/
            dog.1.jpg
            dog.2.jpg
            ...
        cat/
            cat.1.jpg
            cat.2.jpg
            ...

    Parameters
    ----------
    force_download:
        Whether or not force ignore cache
    max_workers:
        Max workers for parallel downloads. Defaults to number of CPUs
    """

    def download(self, root):
        logger = logging.getLogger(__name__)

        logger.info("Downloading ruben/datasets/dogs-vs-cats")
        dataset_dir = SOFIA.download_dataset("ruben/datasets/dogs-vs-cats", force_download=self.force_download,
                                             max_workers=self.max_workers)

        logger.info("Extracting dog.zip ...")
        extract_archive(os.path.join(dataset_dir, "dog.zip"), os.path.join(root, "dog"))
        logger.info("Extracting cat.zip ...")
        extract_archive(os.path.join(dataset_dir, "cat.zip"), os.path.join(root, "cat"))
