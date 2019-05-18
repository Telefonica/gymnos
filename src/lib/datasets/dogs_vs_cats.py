#
#
#   Dogs vs Cats
#
#

import os
import numpy as np

from glob import glob
from tqdm import tqdm

from ..utils.image_utils import imread_rgb, imresize
from .dataset import Dataset, DatasetInfo, Tensor, ClassLabel


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

    def _info(self):
        return DatasetInfo(
            features=Tensor(shape=[150, 150, 3], dtype=np.uint8),
            labels=ClassLabel(names=["cat", "dog"])
        )

    def _download_and_prepare(self, dl_manager):
        train_dir = dl_manager.download_kaggle_and_extract(competition_name="dogs-vs-cats", file_or_files="train.zip")
        cat_images_paths = glob(os.path.join(train_dir, "train", "cat.*.jpg"))
        dog_images_paths = glob(os.path.join(train_dir, "train", "dog.*.jpg"))
        cat_labels = np.full_like(cat_images_paths, 0, dtype=np.int32)
        dog_labels = np.full_like(dog_images_paths, 1, dtype=np.int32)

        images_paths = np.concatenate([cat_images_paths, dog_images_paths], axis=0)
        labels = np.concatenate([cat_labels, dog_labels], axis=0)

        random_indices = np.random.permutation(len(images_paths))

        self.images_paths_ = images_paths[random_indices]
        self.labels_ = labels[random_indices]


    def _load(self):
        images = np.array([self.__read_and_resize_image(image_path) for image_path in tqdm(self.images_paths_)],
                          dtype=np.uint8)
        return images, self.labels_

    def __read_and_resize_image(self, image_path):
        image = imread_rgb(image_path)
        return imresize(image, (150, 150))
