#
#
#   Synthetic digits
#
#

import os
import numpy as np

from glob import glob

from ..utils.image_utils import imread_rgb, imresize
from .dataset import Dataset, DatasetInfo, ClassLabel, Array

KAGGLE_DATASET_NAME = "prasunroy/synthetic-digits"
KAGGLE_DATASET_FILENAME = "data.zip"


class SyntheticDigits(Dataset):
    """
    Synthetic digits with noisy backgrounds for testing robustness of classification algorithms.
    This dataset contains 12,000 synthetically generated images of English digits embedded on random backgrounds.
    The images are generated with varying fonts, colors, scales and rotations.
    The backgrounds are randomly selected from a subset of COCO dataset.
    """

    def info(self):
        return DatasetInfo(
            features=Array(shape=[28, 28, 1], dtype=np.uint8),
            labels=ClassLabel(num_classes=10)
        )

    def download_and_prepare(self, dl_manager):
        data_path = dl_manager.download_kaggle(dataset_name=KAGGLE_DATASET_NAME, file_or_files=KAGGLE_DATASET_FILENAME)
        data_path = dl_manager.extract(data_path)

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
        image = imresize(image, (128, 128))
        return image, label

    def __len__(self):
        return len(self.images_paths_)