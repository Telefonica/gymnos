#
#
#   Unusual Data Usage Test
#
#

import statsmodels.api as sm

from .dataset import LibraryDataset


class UnusualDataUsageTest(LibraryDataset):
    """
    Dataset  of Yearly (1700-2008) data on sunspots from the National Geophysical Data Center.

    Used as test  of unusual data usage models.
    """

    def __init__(self, cache_dir=None):
        super().__init__(cache_dir=None)

    def read(self, download_dir=None):
        data = sm.datasets.sunspots.load_pandas().data
        data = data['SUNACTIVITY'][-13:]
        label_serie = {"pred_last_day": list(data)[-1] * 2, "real_cum_last_day": list(data)[-1]}
        return data[:-1], label_serie
