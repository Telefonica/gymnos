#
#
#   Data Usage Test
#
#

import numpy as np
import statsmodels.api as sm

from .dataset import Dataset, DatasetInfo, Tensor


class DataUsageTest(Dataset):
    """
    Dataset  of Yearly (1700-2008) data on sunspots from the National Geophysical Data Center.

    Used as test  of data usage models.

    Characteristics
        - **Samples total**: xxx
        - **Dimensionality**: xxx
        - **Features**: xxx
    """

    def info(self):
        return DatasetInfo(
            features=Tensor(shape=[], dtype=np.float32),
            labels=Tensor(shape=[], dtype=np.float32)
        )

    def download_and_prepare(self, dl_manager):
        data = sm.datasets.sunspots.load_pandas().data
        data = data['SUNACTIVITY'].values
        # Erase zeros on the left
        consumption_zero = [i for i in list(data) if i > 0.0]
        consumption = np.cumsum(consumption_zero).tolist()
        consumption_zero_acum = [i for i in consumption if i > 0.0]
        consumption_zero_acum = [float(i) for i in consumption_zero_acum]

        self.labels_ = np.array(consumption_zero_acum).reshape(-1, 1)
        self.features_ = np.arange(len(consumption_zero_acum)).reshape(-1, 1)

    def __getitem__(self, index):
        return self.features_[index], self.labels_[index]

    def __len__(self):
        return len(self.features_)
