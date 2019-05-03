#
#
#   File downloader
#
#

import os
import math
import requests

from tqdm import tqdm

from ..logger import get_logger
from ..utils.decompressor import decompress, can_be_decompressed


class FileDownloader:
    """
    Download files from URLS, useful to download public datasets
    """

    def __init__(self):
        self.logger = get_logger(prefix=self)

    def download(self, urls, save_dir=None, unzip=True, verbose=True):
        """
        Download file from URLs.

        Parameters
        ----------
        urls: list or str
            Urls to download.
        save_dir: str, optional
            Directory to save file. If unspecified, current working directory
        unzip: bool, optional
            Whether or not unzip downloaded files.
        verbose: bool, optional
            Whether or not show progress bar
        """
        if save_dir is None:
            self.logger.debug("No saving directory provided. Saving to current directory")
            save_dir = os.getcwd()

        if isinstance(urls, (list, tuple)):
            self.__download_files(urls, save_dir, unzip=unzip, verbose=verbose)
        else:
            self.__download_file(urls, save_dir, unzip=unzip, verbose=verbose)

    def __download_file(self, url, save_dir, unzip=True, verbose=False):
        self.logger.info("Downloading file from url: {}".format(url))
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

        if can_be_decompressed(filename) and unzip:
            self.logger.info("Decompressing {}".format(file_path))
            decompress(file_path)


    def __download_files(self, urls, save_dir, unzip=True, verbose=False):
        for url in urls:
            self.__download_file(url, save_dir, unzip=unzip, verbose=verbose)
