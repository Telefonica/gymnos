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
    Dataset of 50,000 32x32 color training images, labeled over 10 categories, and 10,000 test images.
    """

    def read(self, download_dir=None):
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
        return X, to_categorical(y, 10)
