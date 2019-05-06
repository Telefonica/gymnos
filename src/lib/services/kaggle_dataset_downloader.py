#
#
#   Dataset Downloader
#
#

import os

from ..logger import get_logger

from pydoc import locate
from ..utils.decompressor import decompress, can_be_decompressed


class KaggleDatasetDownloader:
    """
    Download datasets from Kaggle platform
    """

    def __init__(self):
        self.kaggle_api = locate("kaggle.api")

        self.logger = get_logger(prefix=self)

    def download(self, dataset_or_competition_name, filenames=None, save_dir=None, unzip=True, verbose=True):
        """
        Download dataset or competition from Kaggle platform.

        Parameters
        ----------
        dataset_or_competition: str
            Dataset name (``user/dataset-id``) or competition name (``competition-id``)
        filenames: list or str, optional
            Filenames to download
        save_dir: str, optional
            Directory to save data. If unspecified, current working directory.
        unzip: bool, optional
            Whether or not decompress files.
        verbose: bool, optional
            Whether or not show progress bar.
        """
        if save_dir is None:
            self.logger.debug("No saving directory provided. Saving to current directory")
            save_dir = os.getcwd()

        if filenames is None:
            self.__download_whole(dataset_or_competition_name, save_dir, unzip=unzip, verbose=verbose)
        elif isinstance(filenames, (list, tuple)):
            self.__download_files(dataset_or_competition_name, filenames, save_dir, unzip=unzip, verbose=verbose)
        else:
            self.__download_file(dataset_or_competition_name, filenames, save_dir, unzip=unzip, verbose=verbose)

        if unzip:
            for filename in os.listdir(save_dir):
                file_path = os.path.join(save_dir, filename)
                if can_be_decompressed(file_path):
                    decompress(file_path, delete_compressed=False)

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

    def __competition_download_file(self, competition_name, filename, save_dir, unzip=True, verbose=True):
        self.kaggle_api.competition_download_file(competition_name, filename, save_dir, quiet=not verbose)
