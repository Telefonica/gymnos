#
#
#   Boston Housing
#
#

import numpy as np

from .dataset import Dataset, DatasetInfo, Tensor
from keras.datasets import boston_housing


class BostonHousing(Dataset):
    """
    Samples contain 13 attributes of houses at different locations around the Boston suburbs in the late 1970s.
    Targets are the median values of the houses at a location (in k$).

    Characteristics
        - **Samples total**: xxx
        - **Dimensionality**: [13]
        - **Features**: real
    """

    def _info(self):
        return DatasetInfo(
            features=Tensor(shape=[13], dtype=np.float32),
            labels=np.float32
        )

    def _download_and_prepare(self, dl_manager):
        self.logger.debug("Download not required. Using dataset from keras library.")

    def _load(self):
        (X_train, y_train), (X_test, y_test) = boston_housing.load_data()
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
        return X, y
