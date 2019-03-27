#
#
#   Dogs vs Cats
#
#

import os
import numpy as np

from tqdm import tqdm
from glob import iglob
from keras.utils import to_categorical

from .dataset import KaggleDataset
from ..utils.image_utils import imread_rgb, imresize


class DogsVsCats(KaggleDataset):
    """
    Dataset to classify whether images contain either a dog or a cat.
    It contains 25,000 color images of dogs and cats with with an image size of 150x150.
    """

    kaggle_dataset_name = "dogs-vs-cats"
    kaggle_dataset_files = ["train.zip"]


    def read(self, download_dir):
        images_glob = os.path.join(download_dir, "train", "*.jpg")
        images_paths = [f for f in iglob(images_glob) if os.path.isfile(f)]

        images = np.empty([len(images_paths), 150, 150, 3], dtype=np.float)
        classes = np.empty_like(images_paths, dtype=np.int)

        for i, image_path in enumerate(tqdm(images_paths)):
            image = imread_rgb(image_path)
            image = imresize(image, (150, 150))  # resize image because we can't have images with different dimensions
            images[i] = image
            if "dog" in image_path:
                classes[i] = 0
            elif "cat" in image_path:
                classes[i] = 1
            else:
                raise ValueError("Class for image {} not recognized".format(image_path))

        return images, to_categorical(classes, 2)
