#
#
#   CIFAR10
#
#

import numpy as np

from keras.datasets import cifar10

from .dataset import LibraryDataset


class CIFAR10(LibraryDataset):
    """
    Dataset to train computer vision algorithms. It contains color images in 10 different classes.

    The class labels are:

    +----------+--------------+
    | Label    | Description  |
    +==========+==============+
    | 0        | Airplane     |
    +----------+--------------+
    | 1        | Car          |
    +----------+--------------+
    | 2        | Bird         |
    +----------+--------------+
    | 3        | Cat          |
    +----------+--------------+
    | 4        | Deer         |
    +----------+--------------+
    | 5        | Frog         |
    +----------+--------------+
    | 6        | Horse        |
    +----------+--------------+
    | 7        | Cat          |
    +----------+--------------+
    | 8        | Ship         |
    +----------+--------------+
    | 9        | Truck        |
    +----------+--------------+

    Characteristics
        - **Classes**: 10
        - **Samples total**: xxx
        - **Dimensionality**: [32, 32, 3]
        - **Features**: real, between 0 and 255
    """

    def read(self, download_dir):
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
        return X, y
