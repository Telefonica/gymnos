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
        data = data['SUNACTIVITY']
        features_serie = data[:-3].values
        label_serie = np.array(np.array(data).cumsum()[-3:])
        return features_serie, label_serie
