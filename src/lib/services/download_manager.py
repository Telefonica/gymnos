#
#
#   Download Manager
#
#

import os
import uuid
import shutil
import logging

from collections.abc import Iterable

from .kaggle import KaggleService
from ..utils.hashing import sha1_text
from ..utils.text_utils import filenamify_url
from ..utils.extractor import extract_zip, extract_tar, extract_gz
from ..utils.downloader import download_file_from_url

logger = logging.getLogger(__name__)


class DownloadManager:
    """
    Download manager to handle all kinds of download, from URLs to kaggle datasets.

    Parameters
    ----------
    download_dir: str
        Directory to download files. By default, current directory.
    extract_dir: str, optional
        Directory to extract files. By default, <download_dir>/extracted
    force_download: bool, optional
        Whether or not force download if file exists
    force_extraction: bool, optional
        Whether or not force extraction if file exists
    """

    def __init__(self, download_dir=".", extract_dir=None, force_download=False, force_extraction=False):
        self.download_dir = os.path.expanduser(download_dir)
        self.extract_dir = os.path.expanduser(extract_dir or os.path.join(download_dir, "extracted"))
        self.force_download = force_download
        self.force_extraction = force_extraction

        os.makedirs(self.download_dir, exist_ok=True)
        os.makedirs(self.extract_dir, exist_ok=True)


    def extract(self, path_or_paths, ignore_not_compressed=True):
        """
        Extract file/s.

        Parameters
        ----------
        path_or_paths: str or list of str or dict(name: filepath)
            Compressed file paths.
        ignore_not_compressed: bool, optional
            Whether or not raise error if the file is not recognized as compressed file.

        Returns
        --------
        str, list of str or dict
            Directory or directories with extracted files.
            The return type depends on the ``path_or_paths``
            If ``path_or_paths`` is a string, it returns the directory.
            If ``path_or_paths`` is a list of str, it returns a list of directories.
            If ``path_or_paths`` is a dict, it returns a dict(name: directory)
        """
        zip_extensions = (".zip",)
        tar_extensions = (".tar", ".tar.bz2", ".tbz2", ".tbz", ".tb2", ".tar.gz")
        gz_extensions  = (".gz",)
        if isinstance(path_or_paths, str):
            logger.info("Extracting {}".format(path_or_paths))

            basename, extension = os.path.splitext(os.path.basename(path_or_paths))

            if extension in zip_extensions:
                extract_func = extract_zip
            elif extension in tar_extensions:
                extract_func = extract_tar
            elif extension in gz_extensions:
                extract_func = extract_gz
            else:
                if ignore_not_compressed:
                    logger.info("Extension {} not recognized as compressed file. Ignoring".format(extension))
                    return path_or_paths
                else:
                    raise ValueError("Can't extract file {}. Supported extensions: ".format(
                                     path_or_paths, ", ".join(zip_extensions + tar_extensions + gz_extensions)))

            extract_dir = os.path.join(self.extract_dir, basename)
            os.makedirs(extract_dir, exist_ok=True)

            return extract_func(path_or_paths, extract_dir=extract_dir, force=self.force_extraction)
        elif isinstance(path_or_paths, dict):
            data_paths = {}
            for name, path in path_or_paths.items():
                data_paths[name] = self.extract(path)
            return data_paths
        elif isinstance(path_or_paths, Iterable):
            data_paths = []
            for path in path_or_paths:
                download_path = self.extract(path)
                data_paths.append(download_path)
            return data_paths
        else:
            raise ValueError("path_or_paths must be a str, an iterable or a dict. Got {}".format(type(path_or_paths)))


    def download_kaggle(self, dataset_name=None, competition_name=None, file_or_files=None, verbose=True):
        """
        Download kaggle dataset/competition.

        Parameters
        ----------
        dataset_name: str, optional
            Kaggle dataset name with the format <user>/<dataset_name>, e.g mlg-ulb/creditcardfraud.
            Mandatory if ``competition_name`` is None.
        competition_name: str, optional
            Kaggle competition name.
            Mandatory if ``dataset_name`` is None.
        file_or_files: str or list of str or dict(name: filename), optional
            Specific file to download. By default, download all files.
        verbose: bool, optional
            Whether or not show progress bar

        Returns
        -------
        str, list of str or dict
            File paths with downloaded Kaggle files.
            The return type depends on the ``file_or_files`` parameter.
            If ``file_or_files`` is None, it returns the dataset/competition directory.
            If ``file_or_files`` is a str, it returns the directory.
            If ``file_or_files`` is a list of str, it returns a list with the file paths.
            If ``file_or_files`` is a dict, the return type is a dict(name: filepath)
        """
        if file_or_files is None:
            file_or_files = KaggleService.list_files(dataset_name=dataset_name, competition_name=competition_name)

        if isinstance(file_or_files, str):
            if dataset_name is not None:
                logger.info("Downloading {} from kaggle dataset {}".format(file_or_files, dataset_name))
                resource_name = dataset_name.replace("/", "_")
            elif competition_name is not None:
                logger.info("Downloading {} from kaggle competition {}".format(file_or_files, competition_name))
                resource_name = competition_name
            else:
                raise ValueError("You must specify dataset_name or competition_name")

            real_file_path = os.path.join(self.download_dir, resource_name + "_" + file_or_files)

            if os.path.isfile(real_file_path) and not self.force_download:
                if verbose:
                    logger.debug("Download for {}/{} found. Skipping".format(resource_name, file_or_files))
                return real_file_path

            tmp_download_dir = os.path.join(self.download_dir, resource_name + "_" + file_or_files + ".tmp." +
                                            uuid.uuid4().hex)

            KaggleService.download(dataset_name=dataset_name, competition_name=competition_name, files=[file_or_files],
                                   download_dir=tmp_download_dir, force=self.force_download, verbose=verbose)

            logger.info("Removing download temporary directory and moving files")
            shutil.move(os.path.join(tmp_download_dir, file_or_files), real_file_path)
            shutil.rmtree(tmp_download_dir)

            return real_file_path
        elif isinstance(file_or_files, dict):
            file_paths = {}
            for name, filename in file_or_files.items():
                file_paths[name] = self.download_kaggle(dataset_name=dataset_name, competition_name=competition_name,
                                                        file_or_files=filename, verbose=verbose)
            return file_paths
        elif isinstance(file_or_files, Iterable):
            file_paths = []
            for filename in file_or_files:
                file_path = self.download_kaggle(dataset_name=dataset_name, competition_name=competition_name,
                                                 file_or_files=filename, verbose=verbose)
                file_paths.append(file_path)
            return file_paths
        else:
            raise ValueError("file_or_files must be a str, a dict or an iterable")


    def download(self, url_or_urls, verbose=True):
        """
        Download url/s.

        Parameters
        ----------
        url_or_urls: str or list of str or dict(name: url)
            Url or urls to download
        verbose:
            Whether or not show progress bar

        Returns
        -------
        str, list of str or dict
            File paths with downloaded Kaggle files.
            The return type depends on the ``url_or_urls`` parameter.
            If ``url_or_urls`` is a str, it returns the file path.
            If ``url_or_urls`` is a list of str, it returns a list with the file paths.
            If ``url_or_urls`` is a dict, the return type is a dict(name: filepath)
        """
        if isinstance(url_or_urls, str):
            logger.info("Downloading file from url {}".format(url_or_urls))

            sha1_url_hash = sha1_text(url_or_urls)
            slug_url = filenamify_url(url_or_urls)

            if len(slug_url) > 90:
                slug_url = slug_url[:45] + "_" + slug_url[-45:]

            filename = sha1_url_hash + "_" + slug_url

            real_file_path = os.path.join(self.download_dir, filename)

            if os.path.isfile(real_file_path) and not self.force_download:
                if verbose:
                    logger.info("Download for url {} found. Skipping".format(url_or_urls))
                return real_file_path

            tmp_download_dir = os.path.join(self.download_dir, filename + ".tmp." + uuid.uuid4().hex)
            tmp_file_path = os.path.join(tmp_download_dir, filename)

            os.makedirs(tmp_download_dir)

            download_file_from_url(url_or_urls, file_path=tmp_file_path, verbose=verbose,
                                   force=self.force_download)

            logger.info("Removing download temporary directory and moving files")
            shutil.move(tmp_file_path, real_file_path)
            shutil.rmtree(tmp_download_dir)
            return real_file_path
        elif isinstance(url_or_urls, dict):
            file_paths = {}
            for name, url in url_or_urls.items():
                file_paths[name] = self.download(url, verbose=verbose)
            return file_paths
        elif isinstance(url_or_urls, Iterable):
            file_paths = []
            for url in url_or_urls:
                download_path = self.download(url, verbose=verbose)
                file_paths.append(download_path)
            return file_paths
        else:
            raise ValueError("url_or_urls must be a str, an iterable or a dict. Got {}".format(type(url_or_urls)))
