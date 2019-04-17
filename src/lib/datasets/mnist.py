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
    Kind: Classification
    Shape:
        features: [28, 28, 1]
        labels: [10]
    Description: >
        Dataset to predict handwritten images corresponding to numbers between 0-9.
    """

    def read(self, download_dir):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
        return X[..., np.newaxis], to_categorical(y, 10)
