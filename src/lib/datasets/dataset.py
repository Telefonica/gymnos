#
#
#   Dataset
#
#

import os

from keras.utils import to_categorical
from tempfile import TemporaryDirectory

from ..logger import get_logger
from ..utils.hdf_manager import HDFManager


class Dataset:
    """
    Base class for all Gymnos datasets.

    You need to implement the following methods: ``download`` and ``read``.

    Methods
    -------
    download(download_path)
        Download raw data.

        Parameters
        ----------
        download_path: str
            Path where to download data

    read(self, download_path)
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

    def __init__(self, cache_dir=None):
        if cache_dir is not None:
            self.cache = HDFManager(os.path.join(cache_dir, self.__class__.__name__ + ".h5"))
        else:
            self.cache = None

        self.logger = get_logger(prefix=self)


    def load_data(self):
        """
        Check if data exists on cache and download, read and save to cache if not.
        """
        if self.cache is not None and self.cache.exists():
            self.logger.info("Dataset already exists on cache. Retrieving ...")
            X = self.cache.retrieve("X")
            y = self.cache.retrieve("y")

            return X, y

        with TemporaryDirectory() as temp_dir:
            self.logger.info("Downloading dataset ...")
            self.download(temp_dir)

            self.logger.info("Reading dataset ...")
            X, y = self.read(temp_dir)

        if self.cache is not None:
            self.logger.info("Saving dataset to cache ...")
            self.cache.save("X", X)
            self.cache.save("y", y)

        return X, y


class RegressionDataset(Dataset):
    """
    Dataset for regression tasks.
    """


class ClassificationDataset(Dataset):
    """
    Dataset for classification tasks.
    """

    def load_data(self, one_hot=False):
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
        X, y = super().load_data()

        if one_hot:
            y = to_categorical(y)

        return X, y
