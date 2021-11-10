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


@dataclass
class TinySounds(TinySoundsHydraConf, BaseDataset):
    """
    TODO: description about data structure

    Parameters
    -----------
    TODO: description of each parameter
    """

    def download(self, root):
        logger = logging.getLogger(__name__)
        # download_dir = SOFIA.download_dataset(
        #    "IndIAna_jones/datasets/tiny_sounds")

        logger.info("Extracting Sounds ...")
        # extract_archive(os.path.join(download_dir, "Audios.zip"), root)
        print(root)
