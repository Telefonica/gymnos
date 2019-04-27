#
#
# Kaggle Mixin
#
#

from pydoc import locate

from ...logger import get_logger
from ...services.kaggle_dataset_downloader import KaggleDatasetDownloader


class KaggleMixin:
    """
    Mixin to download Kaggle datasets. It provides implementation for ``download`` method.

    Attributes
    ----------
    kaggle_dataset_name: str
        Kaggle dataset or competition to download in the format `username/dataset`.
    kaggle_dataset_files: str or list of str, optional
        Specific files to download (by default whole dataset).
    """

    kaggle_dataset_name = None
    kaggle_dataset_files = None


    def download(self, download_path):
        """
        Download Kaggle dataset/competition specified by ``kaggle_dataset_name``
        and ``kaggle_dataset_files``.

        Parameters
        ----------
        download_path: str
            Path to download kaggle dataset.
        """

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
