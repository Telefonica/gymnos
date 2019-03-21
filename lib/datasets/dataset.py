#
#
#   Dataset
#
#

import os

from tempfile import TemporaryDirectory

from ..logger import logger
from ..services.hdf_manager import HDFManager
from ..services.dataset_downloader import KaggleDatasetDownloader
from ..services.dataset_downloader import PublicDatasetDownloader


class Dataset:

    def __init__(self, cache=None):
        if cache is not None:
            self.cache = HDFManager(os.path.join(cache, self.__class__.__name__ + ".h5"))
        else:
            self.cache = None

    def download(self, download_dir):
        raise NotImplementedError()

    def read(self, download_dir):
        raise NotImplementedError()

    def load_data(self):
        if self.cache is not None and self.cache.exists():
            logger.info("Dataset already exists on cache. Retrieving ...")
            X = self.cache.retrieve("X")
            y = self.cache.retrieve("y")

            return X, y

        with TemporaryDirectory() as temp_dir:
            logger.info("Downloading dataset ...")
            self.download(temp_dir)

            logger.info("Reading dataset ...")
            X, y = self.read(temp_dir)

        if self.cache is not None:
            logger.info("Saving dataset to cache ({}) ...".format(self.cache.file_path))
            self.cache.save("X", X)
            self.cache.save("y", y)

        return X, y


class SparkDataset(Dataset):

    def __init__(self, cache=None):
        super().__init__(cache=None)  # spark dataset is huge so we don't cache data

    def read(self, download_dir=None):
        raise NotImplementedError()

    def load_data(self):
        logger.info("Loading dataset into Spark session ...")
        data = self.read()

        return data


class KaggleDataset(Dataset):

    kaggle_dataset_name = None  # required field
    kaggle_dataset_files = None  # optional field

    def __init__(self, cache=None):
        super().__init__(cache=cache)

        if self.kaggle_dataset_name is None:
            raise ValueError("kaggle_dataset_name cannot be None")

        self.downloader = KaggleDatasetDownloader()

    def download(self, download_dir):
        self.downloader.download(self.kaggle_dataset_name, self.kaggle_dataset_files,
                                 download_dir, verbose=True)


class PublicDataset(Dataset):

    public_dataset_files = None  # required field

    def __init__(self, cache=None):
        super().__init__(cache=None)

        if self.public_dataset_files is None:
            raise ValueError("public_dataset_files cannot be None")

        self.downloader = PublicDatasetDownloader()

    def download(self, download_dir):
        self.downloader.download(self.public_dataset_files, download_dir, verbose=True)


class LibraryDataset(Dataset):

    def read(self, download_dir=None):
        raise NotImplementedError()

    def download(self, download_dir):
        logger.info("Dataset found on Library")
