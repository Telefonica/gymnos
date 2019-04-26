#
#
#   Tiny Imagenet
#
#

import os
import numpy as np

from tqdm import tqdm
from glob import iglob

from .dataset import KaggleDataset
from ..utils.io_utils import read_from_text
from ..utils.image_utils import imread_rgb


class TinyImagenet(KaggleDataset):
    """
    Dataset to classify images. Small version of Imagenet dataset.

    Characteristics
        - **Classes**: 200
        - **Samples total**: xxxx
        - **Dimensionality**: [64, 64, 3]
        - **Features**: real, between 0 and 255
    """

    dataset_name = "tiny_imagenet"
    kaggle_dataset_name = "akash2sharma/tiny-imagenet"

    def read(self, download_dir):
        data_dir = os.path.join(download_dir, "tiny-imagenet-200")

        lines = read_from_text(os.path.join(data_dir, "wnids.txt")).splitlines()
        name2num  = {name: idx for idx, name in enumerate(lines)}

        train_images, classnames = self.__read_train_images(data_dir)
        train_labels = np.array([name2num[classnames[i]] for i in range(len(classnames))])

        test_images, classnames = self.__read_test_images(data_dir)
        test_labels = np.array([name2num[classnames[i]] for i in range(len(classnames))])

        images = np.concatenate([train_images, test_images], axis=0)
        labels = np.concatenate([train_labels, test_labels], axis=0)

        return images, labels

    def __read_train_images(self, data_dir):
        images_glob = os.path.join(data_dir, "train", "**", "images", "*")
        images_list = [f for f in iglob(images_glob) if os.path.isfile(f)]

        images = np.empty([len(images_list), 64, 64, 3], dtype=np.uint8)
        labels = np.empty(len(images_list), dtype=object)

        pbar = tqdm(images_list)
        pbar.set_description("Parsing train images")

        for idx, image_path in enumerate(pbar):
            filename = os.path.basename(image_path)
            class_name = filename.split("_")[0]

            images[idx] = imread_rgb(image_path)
            labels[idx] = class_name

        return images, labels


    def __read_test_images(self, data_dir):
        text = read_from_text(os.path.join(data_dir, "val", "val_annotations.txt"))
        lines = text.splitlines()
        file2classname = {split[0]: split[1] for split in (line.split("\t") for line in lines)}

        images_glob = os.path.join(data_dir, "val", "images", "*")
        images_list = [f for f in iglob(images_glob) if os.path.isfile(f)]

        images = np.empty([len(images_list), 64, 64, 3], dtype=np.uint8)
        labels = np.empty(len(images_list), dtype=object)

        pbar = tqdm(images_list)
        pbar.set_description("Parsing test images")

        for idx, image_path in enumerate(pbar):
            filename = os.path.basename(image_path)
            class_name = file2classname[filename]

            images[idx] = imread_rgb(image_path)
            labels[idx] = class_name

        return images, labels
