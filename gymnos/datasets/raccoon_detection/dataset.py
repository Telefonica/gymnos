#
#
#   Raccoon Detection dataset
#
#

import os
import csv

from ...base import BaseDataset
from ...services.sofia import SOFIA
from ...utils.data_utils import extract_archive
from .hydra_conf import RaccoonDetectionHydraConf

from dataclasses import dataclass
from collections import defaultdict


@dataclass
class RaccoonDetection(RaccoonDetectionHydraConf, BaseDataset):
    """
    TODO: description about data structure

    Parameters
    -----------
    TODO: description of each parameter
    """

    def __call__(self, root):
        download_dir = SOFIA.download_dataset("ruben/datasets/racoon-detection")

        extract_archive(os.path.join(download_dir, "images.zip"), root)

        bboxes_by_fname = defaultdict(list)

        with open(os.path.join(download_dir, "labels.csv"), "r") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                bboxes_by_fname[row["filename"]].append([row["xmin"], row["ymin"], row["xmax"], row["ymax"]])

        with open(os.path.join(root, "labels.txt"), "w") as fp:
            for fname, bboxes in bboxes_by_fname.items():
                line = fname + " "
                for bbox in bboxes:
                    line += ",".join(bbox)
                    line += ",0"  # cls id
                    line += " "
                fp.write(line + "\n")
