#
#
# Kaggle Mixin
#
#

from pydoc import locate

from ...logger import get_logger
from ...services.kaggle_dataset_downloader import KaggleDatasetDownloader


class KaggleMixin:

    kaggle_dataset_name = None
    kaggle_dataset_files = None


    def download(self, download_path):
        if self.kaggle_dataset_name is None:
            raise ValueError("kaggle_dataset_name cannot be None")

        logger = get_logger(prefix=self)
        downloader = KaggleDatasetDownloader()
        kaggle_api_exception = locate("kaggle.rest.ApiException")

        try:
            downloader.download(self.kaggle_dataset_name, self.kaggle_dataset_files,
                                download_path, verbose=True)
        except kaggle_api_exception:
            logger.error("Error downloading Kaggle dataset. Check your credentials.")
            raise
