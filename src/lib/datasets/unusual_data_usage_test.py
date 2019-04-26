#
#
#   Unusual Data Usage Test
#
#

import statsmodels.api as sm

from .dataset import LibraryDataset
import numpy as np


class UnusualDataUsageTest(LibraryDataset):
    """
    Dataset  of Yearly (1700-2008) data on sunspots from the National Geophysical Data Center.

    Used as test  of unusual data usage models.

    Characteristics
        - **Samples total**: xxx
        - **Dimensionality**: xxx
        - **Features**: xxx
    """

    def read(self, download_dir):
        data = sm.datasets.sunspots.load_pandas().data
        label_serie = data['SUNACTIVITY'].values
        features_serie = np.arange(len(label_serie))
        return features_serie, label_serie
