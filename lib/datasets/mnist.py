#
#
#   MNIST
#
#

import numpy as np

from keras.datasets import mnist
from keras.utils import to_categorical

from .dataset import LibraryDataset


class MNIST(LibraryDataset):
    """
    Dataset of 70,000 28x28 grayscale images of the 10 digits.
    """

    def read(self, download_dir=None):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
        return X[..., np.newaxis], to_categorical(y, 10)
