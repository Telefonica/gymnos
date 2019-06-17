#
#
#   Boston Housing
#
#

import logging
import numpy as np

from .dataset import Dataset, DatasetInfo, Array

logger = logging.getLogger(__name__)

DOWNLOAD_URL = "https://s3.amazonaws.com/keras-datasets/boston_housing.npz"


class BostonHousing(Dataset):
    """
    Samples contain 13 attributes of houses at different locations around the Boston suburbs in the late 1970s.
    Targets are the median values of the houses at a location (in k$).

    Characteristics
        - **Samples total**: xxx
        - **Dimensionality**: [13]
        - **Features**: real
    """

    def info(self):
        return DatasetInfo(
            features=Array(shape=[13], dtype=np.float32),
            labels=Array(shape=[], dtype=np.float32)
        )

    def download_and_prepare(self, dl_manager):
        data_path = dl_manager.download(DOWNLOAD_URL)
        logger.info("Loading data")
        self.data_ = np.load(data_path)
        self.size_ = len(self.data_["x"])  # x and y have the same length

    def __getitem__(self, index):
        X = self.data_["x"][index]
        y = self.data_["y"][index]
        return X, y

    def __len__(self):
        return self.size_
