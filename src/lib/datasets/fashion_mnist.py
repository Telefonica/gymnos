#
#
#   Fashion MNIST
#
#

import numpy as np

from .dataset import LibraryDataset

from keras.utils import to_categorical
from keras.datasets import fashion_mnist


class FashionMNIST(LibraryDataset):
    """
    Kind: Classification
    Shape:
        features: [28, 28, 1]
        labels: [10]
    Description: >
        Dataset to classify 10 fashion categories of clothes.
    """

    def read(self, download_dir):
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
        return X, to_categorical(y, 10)
