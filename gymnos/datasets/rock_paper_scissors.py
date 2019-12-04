
#
#
#   RockPaperScissors
#
#

import os
import numpy as np
from glob import glob

from PIL import Image
from ..utils.image_utils import img_to_arr
from .dataset import Dataset, ClassLabel, Array

DATASET_SMB_URI = "smb://10.95.194.112/homes/ruben_salas/rock-paper-scissors.zip"


class RockPaperScissors(Dataset):
    """
    Services: :class:`~gymnos.services.smb.SMB`

    The goal is to predict from grayscale images whether the hand gesture is rock, paper or scissors.
    The class labels are the following:

    +----------+--------------+
    | Label    | Description  |
    +==========+==============+
    | 0        | Rock         |
    +----------+--------------+
    | 1        | Paper        |
    +----------+--------------+
    | 2        | Scissors     |
    +----------+--------------+

    Characteristics:
        - **Classes**: 3
        - **Samples total**: 2641
        - **Dimensionality**: [150, 150, 1]
        - **Features**: real, between 0 and 255
    """

    @property
    def features_info(self):
        return Array(shape=[150, 150, 1], dtype=np.uint8)

    @property
    def labels_info(self):
        return ClassLabel(names=["rock", "paper", "scissors"])

    def download_and_prepare(self, dl_manager):
        path = dl_manager["smb"].download(DATASET_SMB_URI)
        path = dl_manager.extract(path)

        rock_img_paths = glob(os.path.join(path, "rock", "*.png"))
        rock_labels = np.full_like(rock_img_paths, 0, dtype=np.int32)
        paper_img_paths = glob(os.path.join(path, "paper", "*.png"))
        paper_labels = np.full_like(paper_img_paths, 1, dtype=np.int32)
        scissors_img_paths = glob(os.path.join(path, "scissors", "*.png"))
        scissors_labels = np.full_like(scissors_img_paths, 2, dtype=np.int32)

        self.images_paths_ = np.concatenate([rock_img_paths, paper_img_paths, scissors_img_paths], axis=0)
        self.labels_ = np.concatenate([rock_labels, paper_labels, scissors_labels], axis=0)

    def __getitem__(self, index):
        image = Image.open(self.images_paths_[index]).convert("L")
        image = image.resize((150, 150))
        image = img_to_arr(image)
        return image, self.labels_[index]

    def __len__(self):
        return len(self.images_paths_)
