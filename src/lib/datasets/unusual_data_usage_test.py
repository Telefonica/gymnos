#
#
#   Unusual Data Usage Test
#
#

import numpy as np
import statsmodels.api as sm

from .dataset import Dataset, DatasetInfo


class UnusualDataUsageTest(Dataset):
    """
    Dataset  of Yearly (1700-2008) data on sunspots from the National Geophysical Data Center.

    Used as test  of unusual data usage models.

    Characteristics
        - **Samples total**: xxx
        - **Dimensionality**: xxx
        - **Features**: xxx
    """

    def _info(self):
        return DatasetInfo(
            features=np.int32,
            labels=np.float32
        )

    def _download_and_prepare(self, dl_manager):
        print("Download not required. Using dataset from statsmodels library.")

    def _load(self):
        data = sm.datasets.sunspots.load_pandas().data
        label_serie = data['SUNACTIVITY'].values
        features_serie = np.arange(len(label_serie))
        return features_serie, label_serie
