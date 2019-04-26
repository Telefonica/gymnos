#
#
#   Fashion MNIST
#
#

import numpy as np

from .dataset import LibraryDataset

from keras.datasets import fashion_mnist


class FashionMNIST(LibraryDataset):
    """
    Dataset to classify 10 fashion categories of clothes..

    The class labels are:

    +----------+--------------+
    | Label    | Description  |
    +==========+==============+
    | 0        | T-shirt/top  |
    +----------+--------------+
    | 1        | Trouser      |
    +----------+--------------+
    | 2        | Pullover     |
    +----------+--------------+
    | 3        | Dress        |
    +----------+--------------+
    | 4        | Coat         |
    +----------+--------------+
    | 5        | Sandal       |
    +----------+--------------+
    | 6        | Shirt        |
    +----------+--------------+
    | 7        | Sneaker      |
    +----------+--------------+
    | 8        | Bag          |
    +----------+--------------+
    | 9        | Ankle boot   |
    +----------+--------------+

    Characteristics
        - **Classes**: 10
        - **Samples total**: 70 000
        - **Dimensionality**: [28, 28, 1]
        - **Features**: real, between 0 and 255
    """

    def read(self, download_dir):
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
        return X, y
