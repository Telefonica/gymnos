#
#
#   Paintings dataset
#
#
import logging
import os
import zipfile


from ...base import BaseDataset
from ...services.sofia import SOFIA
from .hydra_conf import PaintingsHydraConf

from dataclasses import dataclass


@dataclass
class Paintings(PaintingsHydraConf, BaseDataset):

    def download(self, root):
        
        logger = logging.getLogger(__name__)

        download_dir = SOFIA.download_dataset("alvaromarcom/datasets/pandorIA_dataset0", files=["archive_5.zip"])

        logger.info("Extracting archive_5.zip ...")

        with zipfile.ZipFile(os.path.join(download_dir, "archive_5.zip")) as zip:
            for zip_info in zip.infolist():
                if zip_info.filename[-1] == '/':
                    continue

                zip_info.filename = os.path.basename(zip_info.filename)

                if not self.force_extract and os.path.isfile(os.path.join(root, zip_info.filename)):
                    continue

                zip.extract(zip_info, root)
