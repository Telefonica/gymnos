#
#
#   Dataset
#
#

import os

from pydoc import locate
from tempfile import TemporaryDirectory

from ..logger import get_logger
from ..utils.hdf_manager import HDFManager
from ..services.kaggle_dataset_downloader import KaggleDatasetDownloader
from ..services.file_downloader import FileDownloader


class Dataset:

    def __init__(self, cache_dir=None):
        if cache_dir is not None:
            self.cache = HDFManager(os.path.join(cache_dir, self.__class__.__name__ + ".h5"))
        else:
            self.cache = None

        self.logger = get_logger(prefix=self)

    def download(self, download_dir):
        # Implement to give instructions about how to download the dataset.
        # The dataset will be stored in a temporary directory.
        raise NotImplementedError()

    def read(self, download_dir):
        # Implement to give instructions about how to read the dataset from the download directory.
        # It must return X (features) and y (labels).
        # Labels must be one-hot encoded.
        raise NotImplementedError()

    def load_data(self):
        # Download data, read data, save data to cache if defined, return data (features and labels).
        # If data exists on cache, retrieve data from cache.
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


class KaggleDataset(Dataset):

    """
    Parent class for all Kaggle datasets.
    Must implement:
        - read(download_dir)
    """

    kaggle_dataset_name = None  # required field
    kaggle_dataset_files = None  # optional field | specify only if we want to download specific files

    def __init__(self, cache_dir=None):
        super().__init__(cache_dir=cache_dir)

        if self.kaggle_dataset_name is None:
            raise ValueError("kaggle_dataset_name cannot be None")

        self.logger = get_logger(prefix=self)

        self.downloader = KaggleDatasetDownloader()

    def download(self, download_dir):
        kaggle_api_exception = locate("kaggle.rest.ApiException")

        try:
            self.downloader.download(self.kaggle_dataset_name, self.kaggle_dataset_files,
                                     download_dir, verbose=True)
        except kaggle_api_exception as exception:
            self.logger.error("Error downloading Kaggle dataset. Check your credentials.")
            raise exception


class PublicDataset(Dataset):

    """
    Parent class for all public datasets available with a URL.
    Must implement:
        - read(download_dir)
    """

    public_dataset_files = None  # required field | urls of files to download

    def __init__(self, cache_dir=None):
        super().__init__(cache_dir=cache_dir)

        if self.public_dataset_files is None:
            raise ValueError("public_dataset_files cannot be None")

        self.downloader = FileDownloader()

    def download(self, download_dir):
        self.downloader.download(self.public_dataset_files, download_dir, verbose=True)


class LibraryDataset(Dataset):

    """
    Parent class for all datasets available with a library e.g Keras, Scikit-Learn, NLTK, etc ...
    Must implement:
        - read(download_dir)
    """

    def read(self, download_dir=None):
        raise NotImplementedError()

    def download(self, download_dir):
        # Download is handled by library
        self.logger.info("Retrieving dataset from library ...")
