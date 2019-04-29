#
#
#   Dataset
#
#

import os
import shutil
import tempfile

from keras.utils import to_categorical

from ..logger import get_logger
from ..utils.hdf_manager import HDFManager


class Dataset:
    """
    Base class for all Gymnos datasets.

    You need to implement the following methods: ``download`` and ``read``.
    """

    def __init__(self):
        self.logger = get_logger(prefix=self)


    def download(self, download_path):
        """
        Download raw data.

        Parameters
        ----------
        download_path: str
            Path where to download data
        """
        return super().download(download_path)


    def read(self, download_path):
        """
        Read data from download path

        Parameters
        ----------
            download_path: str
                Path where to read data
        Returns
        -------
            X: array_like
                Features
            y: array_like
                Labels
        """
        return super().read(download_path)

    def load_data(self, hdf5_cache_path=None):
        """
        Check if data exists on cache and download, read and save to cache if not.

        Returns
        -------
        X: array_like
            Features
        y: array_like
            Labels
        """

        if hdf5_cache_path is not None:
            cache = HDFManager(hdf5_cache_path)
        else:
            cache = None

        if cache is not None and cache.exists():
            return cache.retrieve("X"), cache.retrieve("y")

        gymnos_dataset_temp_path = os.path.join(tempfile.gettempdir(), "gymnos", self.__class__.__name__)

        self.logger.info("Checking if download exists in temporary directory ({})".format(gymnos_dataset_temp_path))

        if not os.path.isdir(gymnos_dataset_temp_path):
            self.logger.info("Download not found. Creating download directory.")
            os.makedirs(gymnos_dataset_temp_path)
            try:
                self.logger.info("Downloading")
                self.download(gymnos_dataset_temp_path)
            except Exception as ex:
                self.logger.error("Error downloading dataset. Removing download directory")
                shutil.rmtree(gymnos_dataset_temp_path)
                raise
        else:
            self.logger.info("Data exists in temporary directory")

        self.logger.info("Reading from {}".format(gymnos_dataset_temp_path))
        X, y = self.read(gymnos_dataset_temp_path)

        if cache is not None:
            cache.save("X", X)
            cache.save("y", y)

        return X, y


class RegressionDataset(Dataset):
    """
    Dataset for regression tasks.

    You need to implement the following methods: ``download`` and ``read``.
    """


class ClassificationDataset(Dataset):
    """
    Dataset for classification tasks.

    You need to implement the following methods: ``download`` and ``read``.
    """

    def load_data(self, one_hot=False, hdf5_cache_path=None):
        """
        Check if data exists on cache and download, read and save to cache if not.

        Parameters
        ----------
        one_hot: bool, optional
            Whether or note one-hot encode labels.
        Returns
        -------
        X: array_like
            Features.
        y: array_like of int
            Labels.
        """
        X, y = super().load_data(hdf5_cache_path=hdf5_cache_path)

        if one_hot:
            y = to_categorical(y)

        return X, y
