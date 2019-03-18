#
#
#   Dataset Downloader
#
#

import os
import math
import requests

from tqdm import tqdm
from kaggle import api
from ..utils.decompressor import Decompressor


class KaggleDatasetDownloader:
    """
    Download datasets from Kaggle platform
    """

    def __init__(self):
        has_username = "KAGGLE_USERNAME" in os.environ
        has_api_key = "KAGGLE_KEY" in os.environ

        if not has_username or not has_api_key:
            msg  = "Environment variables for Kaggle API not found."
            msg += "You need to provide KAGGLE_USERNAME and KAGGLE_KEY"
            raise Exception(msg)

    def download(self, dataset_or_competition_name, filenames=None, save_dir=None, unzip=True, verbose=True):
        if save_dir is None:
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
        if self.__is_a_competition(dataset_name):
            self.__competition_download(dataset_name, save_dir, unzip=unzip, verbose=verbose)
        else:
            self.__dataset_download(dataset_name, save_dir, unzip=unzip, verbose=verbose)

    def __dataset_download(self, dataset_name, save_dir, unzip=True, verbose=True):
        api.dataset_download_files(dataset_name, path=save_dir, unzip=unzip, quiet=not verbose)

    def __competition_download(self, competition_name, save_dir, unzip=True, verbose=True):
        api.competition_download_files(competition_name, path=save_dir, unzip=unzip, quiet=not verbose)

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

    def __download_file(self, dataset_name, filename, save_dir, unzip=True, verbose=False):
        if self.__is_a_competition(dataset_name):
            self.__competition_download_file(dataset_name, filename, save_dir, unzip=True, verbose=False)
        else:
            self.dataset_download_file(dataset_name, filename, save_dir, unzip=True, verbose=False)

    def __dataset_download_file(self, dataset_name, filename, save_dir, unzip=True, verbose=False):
        api.dataset_download_file(dataset_name, filename, save_dir, quiet=not verbose)
        self.__unzip_and_delete_if_needed(os.path.join(save_dir, filename))

    def __competition_download_file(self, competition_name, filename, save_dir, unzip=True, verbose=False):
        api.competition_download_file(competition_name, filename, save_dir, quiet=not verbose)
        self.__unzip_and_delete_if_needed(os.path.join(save_dir, filename))

    def __unzip_and_delete_if_needed(self, file_path):
        file_exists = os.path.isfile(file_path)

        if file_exists:
            Decompressor.extract(file_path, delete_compressed=True)


class PublicDatasetDownloader:
    """
    Download Datasets from public URLs
    """

    def download(self,  urls, save_dir=None, unzip=True, verbose=True):
        if save_dir is None:
            save_dir = os.getcwd()

        if isinstance(urls, (list, tuple)):
            self.__download_files(urls, save_dir, unzip=unzip, verbose=verbose)
        else:
            self.__download_file(urls, save_dir, unzip=unzip, verbose=verbose)

    def __download_file(self, url, save_dir, unzip=True, verbose=False):
        filename = os.path.basename(url)
        file_path = os.path.join(save_dir, filename)
        with requests.get(url, stream=True) as r, open(file_path, "wb") as f:
            block_size = 1024
            total_size = int(r.headers.get('content-length', 0))

            iterator = r.iter_content(block_size)

            if verbose:
                num_blocks = math.ceil(total_size // block_size)
                iterator = tqdm(iterator, total=num_blocks, unit="KB", unit_scale=True)

            for data in iterator:
                f.write(data)

        if Decompressor.is_compressed(filename):
            Decompressor.extract(file_path)


    def __download_files(self, urls, save_dir, unzip=True, verbose=False):
        for url in urls:
            self.download_file(url, save_dir, unzip=unzip, verbose=verbose)
