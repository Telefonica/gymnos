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
    Dataset of 70,000 28x28 grayscale images of 10 fashion categories.
    """

    def read(self, download_dir=None):
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
        return X, to_categorical(y, 10)
