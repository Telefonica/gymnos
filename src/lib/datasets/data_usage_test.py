#
#
#   Data Usage Test
#
#

import numpy as np
import statsmodels.api as sm

from .dataset import LibraryDataset


class DataUsageTest(LibraryDataset):
    """
    Dataset  of Yearly (1700-2008) data on sunspots from the National Geophysical Data Center.

    Used as test  of data usage models.
    """

    def __init__(self, cache_dir=None):
        super().__init__(cache_dir=None)

    def read(self, download_dir=None):
        data = sm.datasets.sunspots.load_pandas().data
        data = data['SUNACTIVITY'].values
        # Erase zeros on the left
        consumption_zero = [i for i in list(data) if i > 0.0]
        consumption = np.cumsum(consumption_zero).tolist()
        consumption_zero_acum = [i for i in consumption if i > 0.0]
        consumption_zero_acum = [float(i) for i in consumption_zero_acum]

        label_serie = np.array(consumption_zero_acum).reshape(-1,1)
        features_serie = np.arange(len(consumption_zero_acum)).reshape(-1,1)
        return features_serie, label_serie