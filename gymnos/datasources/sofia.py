#
#
#   SOFIA data source
#
#

from ..services.sofia import SOFIA


def SOFIADataSource(dataset, files=None, force_download=False, max_workers=None):
    return SOFIA.download_dataset(dataset, files, force_download, max_workers)
