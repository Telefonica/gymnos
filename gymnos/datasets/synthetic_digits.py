#
#
#   Synthetic digits
#
#

import os
import numpy as np

from glob import glob

from .dataset import Dataset, ClassLabel, Array
from ..preprocessors.utils.image_ops import imread_rgb, imresize

KAGGLE_DATASET_NAME = "prasunroy/synthetic-digits"

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_DEPTH = 3


class SyntheticDigits(Dataset):
    """
    Services: :class:`~gymnos.services.kaggle.Kaggle`

    Synthetic digits with noisy backgrounds for testing robustness of classification algorithms.
    This dataset contains 12,000 synthetically generated images of English digits embedded on random backgrounds.
    The images are generated with varying fonts, colors, scales and rotations.
    The backgrounds are randomly selected from a subset of COCO dataset.

    Characteristics:
        - **Classes**: 3
        - **Samples total**: 2641
        - **Dimensionality**: [150, 150, 1]
        - **Features**: real, between 0 and 255
    """

    @property
    def features_info(self):
        return Array(shape=[IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH], dtype=np.uint8)

    @property
    def labels_info(self):
        return ClassLabel(num_classes=10)

    def download_and_prepare(self, dl_manager):
        data_path = dl_manager["kaggle"].download(dataset_name=KAGGLE_DATASET_NAME)

        train_imgs_path = os.path.join(data_path, "synthetic_digits", "imgs_train")

        self.images_paths_ = []
        self.labels_ = []

        for dir_num in range(9):
            class_images_paths = glob(os.path.join(train_imgs_path, str(dir_num), "*.jpg"))
            class_labels = [dir_num] * len(class_images_paths)

            self.labels_.extend(class_labels)
            self.images_paths_.extend(class_images_paths)

    def __getitem__(self, index):
        label = self.labels_[index]
        image = imread_rgb(self.images_paths_[index])
        image = imresize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        return image, label

    def __len__(self):
        return len(self.images_paths_)
