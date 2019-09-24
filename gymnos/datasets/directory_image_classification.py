#
#
#   Image Classification Generator
#
#

import os
import logging
import numpy as np

from PIL import Image
from urllib.parse import urlparse

from .dataset import Dataset, Array, ClassLabel
from ..utils.image_utils import img_to_arr

logger = logging.getLogger(__name__)

IMAGE_WHITE_LISTS_FORMATS = ("png", "jpg", "jpeg", "bmp", "ppm", "tif", "tiff")

# MARK: Helpers


def _iter_valid_files(directory, white_list_formats):
    """
    Iterates on files with extension in `white_list_formats` contained in `directory`.

    Parameters
    ------------
        directory: str
            Absolute path to the directory containing files to be counted
        white_list_formats: list of str
            Set of strings containing allowed extensions for the files to be counted.
    Yields
    --------
        File path with extension in `white_list_formats`.
    """
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.lower().endswith(white_list_formats):
                yield os.path.join(root, filename)


class DirectoryImageClassification(Dataset):
    """
    Simple dataset to classify images from a directory.
    Images for each class must be in a directory with the class name.
    An example of this structure is the following:

    .. code-block::

        path/
            class_name_1/
                img_1.jpg
                img_2.jpg
                img_3.jpg
            class_name_2/
                img_4.jpg
                img_5.jpg
                img_6.jpg
            class_name_3/
                img_7.jpg
                img_8.jpg
                img_9.jpg

    Parameters
    ===========
    path: str
        Directory or compressed file path where each subdirectory contains the images for the class.
    size: 2-tuple, optional
        The requested size in pixels, as a 2-tuple: `(width, height)`. By default, (256, 256).
    color_mode: str, optional
        One of "grayscale" or "rgb". Whether the images will be converted to have 1 or 3 color channels.
    """

    def __init__(self, path, size=None, color_mode="rgb"):
        assert color_mode in ("grayscale", "rgb")

        if size is None:
            size = (256, 256)

        self.path = path
        self.size = size
        self.color_mode = color_mode

    @property
    def features_info(self):
        if self.color_mode == "grayscale":
            depth = 1
        elif self.color_mode == "rgb":
            depth = 3

        return Array(shape=list(self.size) + [depth], dtype=np.uint8)

    @property
    def labels_info(self):
        return ClassLabel(names=self.class_names_)

    def download_and_prepare(self, dl_manager):
        parsed_path = urlparse(self.path)

        if parsed_path.scheme == "smb":
            local_path = dl_manager["smb"].download(self.path)
        else:
            local_path = self.path

        local_path = dl_manager.extract(local_path)  # maybe is a compressed file

        class_directories = [f.path for f in os.scandir(local_path) if f.is_dir()]

        self.class_names_ = [os.path.basename(class_dir) for class_dir in class_directories]

        self.images_paths_ = []
        self.images_labels_ = []

        for class_idx, class_directory in enumerate(class_directories):
            for image_path in _iter_valid_files(class_directory, IMAGE_WHITE_LISTS_FORMATS):
                self.images_paths_.append(image_path)
                self.images_labels_.append(class_idx)

        logger.info("Found {} images belonging to {} classes".format(len(self.images_paths_), len(class_directories)))

    def __getitem__(self, index):
        image_path, image_label = self.images_paths_[index], self.images_labels_[index]

        image = Image.open(image_path)
        image = image.resize(self.size)

        if self.color_mode == "grayscale":
            image = image.convert("L")

        return img_to_arr(image), image_label

    def __len__(self):
        return len(self.images_paths_)
