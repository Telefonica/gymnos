#
#
#   CIFAR10
#
#

import numpy as np

from keras.datasets import cifar10
from keras.utils import to_categorical

from .dataset import LibraryDataset


class CIFAR10(LibraryDataset):
    """
    Kind: Classification
    Shape:
        features: [32, 32, 3]
        labels: [10]
    Description: >
        Dataset to train computer vision algorithms. It contains color images in 10 different classes.
        The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks.
    """

    def read(self, download_dir=None):
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
        return X, to_categorical(y, 10)
