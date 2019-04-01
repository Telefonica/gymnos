#
#
#   Dataset Downloader
#
#

import os

from ..logger import get_logger

from pydoc import locate
from ..utils.decompressor import decompress


class KaggleDatasetDownloader:
    """
    Download datasets from Kaggle platform
    """

    def __init__(self):
        has_username = os.getenv("KAGGLE_USERNAME", False)
        has_api_key = os.getenv("KAGGLE_KEY", False)

        if not has_username or not has_api_key:
            msg  = "Environment variables for Kaggle API not found."
            msg += "You need to provide KAGGLE_USERNAME and KAGGLE_KEY to download a Kaggle dataset"
            raise Exception(msg)

        self.kaggle_api = locate("kaggle.api")

        self.logger = get_logger(prefix=self)

    def download(self, dataset_or_competition_name, filenames=None, save_dir=None, unzip=True, verbose=True):
        if save_dir is None:
            self.logger.debug("No saving directory provided. Saving to current directory")
            save_dir = os.getcwd()

        if filenames is None:
            self.__download_whole(dataset_or_competition_name, save_dir, unzip=unzip, verbose=verbose)
        elif isinstance(filenames, (list, tuple)):
            self.__download_files(dataset_or_competition_name, filenames, save_dir, unzip=unzip, verbose=verbose)
        else:
            self.__download_file(dataset_or_competition_name, filenames, save_dir, unzip=unzip, verbose=verbose)

    def __is_a_competition(self, dataset_name):
        return "/" not in dataset_name

    def __download_whole(self, dataset_name, save_dir, unzip=True, verbose=True):
        self.logger.info("Downloading from {}".format(dataset_name))
        if self.__is_a_competition(dataset_name):
            self.__competition_download(dataset_name, save_dir, unzip=unzip, verbose=verbose)
        else:
            self.__dataset_download(dataset_name, save_dir, unzip=unzip, verbose=verbose)

    def __dataset_download(self, dataset_name, save_dir, unzip=True, verbose=True):
        self.kaggle_api.dataset_download_files(dataset_name, path=save_dir, unzip=unzip, quiet=not verbose)

    def __competition_download(self, competition_name, save_dir, unzip=True, verbose=True):
        self.kaggle_api.competition_download_files(competition_name, path=save_dir, unzip=unzip, quiet=not verbose)

    def __download_files(self, dataset_name, filenames, save_dir, unzip=True, verbose=True):
        if self.__is_a_competition(dataset_name):
            self.__competition_download_files(dataset_name, filenames, save_dir, unzip=True, verbose=True)
        else:
            self.__dataset_download_files(dataset_name, filenames, save_dir, unzip=True, verbose=True)

    def __dataset_download_files(self, dataset_name, filenames, save_dir, unzip=True, verbose=True):
        for filename in filenames:
            self.__dataset_download_file(dataset_name, filename, save_dir, unzip, verbose)

    def __competition_download_files(self, competition_name, filenames, save_dir, unzip=True, verbose=True):
        for filename in filenames:
            self.__competition_download_file(competition_name, filename, save_dir, unzip, verbose)

    def __download_file(self, dataset_name, filename, save_dir, unzip=True, verbose=True):
        self.logger.info("Downloading {} from {}".format(filename, dataset_name))
        if self.__is_a_competition(dataset_name):
            self.__competition_download_file(dataset_name, filename, save_dir, unzip=True, verbose=True)
        else:
            self.dataset_download_file(dataset_name, filename, save_dir, unzip=True, verbose=True)

    def __dataset_download_file(self, dataset_name, filename, save_dir, unzip=True, verbose=True):
        self.kaggle_api.dataset_download_file(dataset_name, filename, save_dir, quiet=not verbose)
        self.__unzip_and_delete_if_needed(os.path.join(save_dir, filename))

    def __competition_download_file(self, competition_name, filename, save_dir, unzip=True, verbose=True):
        self.kaggle_api.competition_download_file(competition_name, filename, save_dir, quiet=not verbose)
        self.__unzip_and_delete_if_needed(os.path.join(save_dir, filename))

    def __unzip_and_delete_if_needed(self, file_path):
        file_exists = os.path.isfile(file_path)

        if file_exists:
            decompress(file_path, delete_compressed=True)
