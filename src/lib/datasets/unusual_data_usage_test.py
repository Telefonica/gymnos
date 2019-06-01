#
#
#   Unusual Data Usage Test
#
#

import numpy as np
import statsmodels.api as sm

from .dataset import Dataset, DatasetInfo, Array


class UnusualDataUsageTest(Dataset):
    """
    Dataset  of Yearly (1700-2008) data on sunspots from the National Geophysical Data Center.

    Used as test  of unusual data usage models.

    Characteristics
        - **Samples total**: xxx
        - **Dimensionality**: xxx
        - **Features**: xxx
    """

    def info(self):
        return DatasetInfo(
            features=Array(shape=[], dtype=np.int64),
            labels=Array(shape=[], dtype=np.float64)
        )

    def download_and_prepare(self, dl_manager):
        data = sm.datasets.sunspots.load_pandas().data
        self.labels_ = data['SUNACTIVITY'].values
        self.features_ = np.arange(len(self.labels_))

    def __getitem__(self, index):
        return self.features_[index], self.labels_[index]

    def __len__(self):
        return len(self.features_)
