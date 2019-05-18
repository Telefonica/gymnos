#
#
#   MNIST
#
#

import numpy as np

from keras.datasets import mnist

from .dataset import Dataset, Tensor, ClassLabel, DatasetInfo


class MNIST(Dataset):
    """
    Dataset to predict handwritten images corresponding to numbers between 0-9.

    Characteristics
        - **Classes**: 10
        - **Samples total**: 70 000
        - **Dimensionality**: [28, 28, 1]
        - **Features**: real, between 0 and 255
    """

    def _info(self):
        return DatasetInfo(
            features=Tensor(shape=[28, 28, 1], dtype=np.uint8),
            labels=ClassLabel(num_classes=10)
        )

    def _download_and_prepare(self, dl_manager):
        print("Download not required. Using dataset from keras library.")

    def _load(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
        return X[..., np.newaxis], y
