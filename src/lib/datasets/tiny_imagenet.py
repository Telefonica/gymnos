#
#
#   Tiny Imagenet
#
#

import os
import numpy as np

from glob import glob

from .dataset import Dataset, DatasetInfo, Array, ClassLabel
from ..utils.io_utils import read_from_text
from ..utils.image_utils import imread_rgb


class TinyImagenet(Dataset):
    """
    Dataset to classify images. Small version of Imagenet dataset.

    Characteristics
        - **Classes**: 200
        - **Samples total**: xxxx
        - **Dimensionality**: [64, 64, 3]
        - **Features**: real, between 0 and 255
    """

    def info(self):
        return DatasetInfo(
            features=Array(shape=[64, 64, 3], dtype=np.uint8),
            labels=ClassLabel(names_file=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                      "tiny_imagenet_labels.txt"))
        )

    def download_and_prepare(self, dl_manager):
        path = dl_manager.download_kaggle(dataset_name="akash2sharma/tiny-imagenet",
                                          file_or_files="tiny-imagenet-200.zip")
        path = dl_manager.extract(path)
        path = os.path.join(path, "tiny-imagenet-200")

        lines = read_from_text(os.path.join(path, "wnids.txt")).splitlines()
        name2num  = {name: idx for idx, name in enumerate(lines)}

        train_images_paths = glob(os.path.join(path, "train", "**", "images", "*.JPEG"))
        train_classnames = [os.path.basename(image_path).split("_")[0] for image_path in train_images_paths]
        train_labels = np.array([name2num[classname] for classname in train_classnames], dtype=np.int32)

        val_names = read_from_text(os.path.join(path, "val", "val_annotations.txt"))
        valfile2classname = {split[0]: split[1] for split in (line.split("\t") for line in val_names.splitlines())}
        val_images_paths = glob(os.path.join(path, "val", "images", "*.JPEG"))
        val_classnames = [valfile2classname[os.path.basename(filename)] for filename in val_images_paths]
        val_labels = np.array([name2num[classname] for classname in val_classnames], dtype=np.int32)

        labels = np.concatenate([train_labels, val_labels], axis=0)
        images_paths = np.concatenate([train_images_paths, val_images_paths], axis=0)

        random_indices = np.random.permutation(len(images_paths))

        self.labels_ = labels[random_indices]
        self.images_paths_ = images_paths[random_indices]


    def __getitem__(self, index):
        image = imread_rgb(self.images_paths_[index])
        return image, self.labels_[index]


    def __len__(self):
        return len(self.images_paths_)
