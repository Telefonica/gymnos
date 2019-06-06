#
#
#   Dogs vs Cats
#
#

import os
import logging
import numpy as np

from glob import glob

from ..utils.image_utils import imread_rgb, imresize
from .dataset import Dataset, DatasetInfo, Array, ClassLabel

IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150
IMAGE_DEPTH = 3

KAGGLE_COMPETITION_NAME = "dogs-vs-cats"
KAGGLE_COMPETITION_FILE = "train.zip"

logger = logging.getLogger(__name__)


class DogsVsCats(Dataset):
    """
    Dataset to classify whether images contain either a dog or a cat.

    The class labels are:

    +----------+--------------+
    | Label    | Description  |
    +==========+==============+
    | 0        | Dog          |
    +----------+--------------+
    | 1        | Cat          |
    +----------+--------------+

    Characteristics
        - **Classes**: 2
        - **Samples total**: xxx
        - **Dimensionality**: [150, 150, 3]
        - **Features**: real, between 0 and 255
    """

    def info(self):
        return DatasetInfo(
            features=Array(shape=[IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH], dtype=np.uint8),
            labels=ClassLabel(names=["cat", "dog"])
        )


    def download_and_prepare(self, dl_manager):
        train_files = dl_manager.download_kaggle(competition_name=KAGGLE_COMPETITION_NAME,
                                                 file_or_files=KAGGLE_COMPETITION_FILE)
        train_dir = dl_manager.extract(train_files)
        cat_images_paths = glob(os.path.join(train_dir, "train", "cat.*.jpg"))
        dog_images_paths = glob(os.path.join(train_dir, "train", "dog.*.jpg"))
        cat_labels = np.full_like(cat_images_paths, 0, dtype=np.int32)
        dog_labels = np.full_like(dog_images_paths, 1, dtype=np.int32)

        images_paths = np.concatenate([cat_images_paths, dog_images_paths], axis=0)
        labels = np.concatenate([cat_labels, dog_labels], axis=0)

        random_indices = np.random.permutation(len(images_paths))

        self.images_paths_ = images_paths[random_indices]
        self.labels_ = labels[random_indices]


    def __getitem__(self, index):
        image = imread_rgb(self.images_paths_[index])
        image = imresize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        return image, self.labels_[index]


    def __len__(self):
        return len(self.images_paths_)
