#
#
#   Covid19ChestXray
#
#

import os
import glob
import numpy as np

from .dataset import Dataset, Array, ClassLabel
from ..preprocessors.utils.image_ops import imread_rgb, imresize


class Covid19ChestXray(Dataset):
    """
    TODO(Covid19ChestXray): Description of my dataset.
    """

    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size

    @property
    def features_info(self):
        return Array(shape=[*self.target_size, 3], dtype=np.uint8)

    @property
    def labels_info(self):
        return ClassLabel(names=["Normal", "Covid-19", "Viral-Pneumonia"])

    def download_and_prepare(self, dl_manager):
        download_dir = dl_manager["kaggle"].download(dataset_name="tawsifurrahman/covid19-radiography-database")

        dataset_dir = os.path.join(download_dir, "COVID-19 Radiography Database")

        normal_imgs_fpaths = glob.glob(os.path.join(dataset_dir, "NORMAL", "*.png"))
        covid19_imgs_fpaths = glob.glob(os.path.join(dataset_dir, "COVID-19", "*.png"))
        pneumonia_imgs_fpaths = glob.glob(os.path.join(dataset_dir, "Viral Pneumonia", "*.png"))

        self.images = np.concatenate([normal_imgs_fpaths, covid19_imgs_fpaths, pneumonia_imgs_fpaths])
        self.targets = np.concatenate([
            0 * np.ones_like(normal_imgs_fpaths, int),
            1 * np.ones_like(covid19_imgs_fpaths, int),
            2 * np.ones_like(pneumonia_imgs_fpaths, int)
        ])

    def __getitem__(self, index):
        image_path, target = self.images[index], self.targets[index]
        image = imread_rgb(image_path)
        image = imresize(image, self.target_size)
        return image, target

    def __len__(self):
        return len(self.images)
