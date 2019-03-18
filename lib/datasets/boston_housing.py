#
#
#   Boston Housing
#
#

import numpy as np

from .dataset import LibraryDataset

from keras.datasets import boston_housing


class BostonHousing(LibraryDataset):
    """
    Samples contain 13 attributes of houses at different locations around the Boston suburbs in the late 1970s.
    Targets are the median values of the houses at a location (in k$).
    """

    def read(self, download_dir):
        (X_train, y_train), (X_test, y_test) = boston_housing.load_data()
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
        return X, y
