#
#
#   Download Manager
#
#

import os
import logging

from collections.abc import Iterable

from ..utils.archiver import extract_zip, extract_tar, extract_gz

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

    def __init__(self, download_dir="downloads", extract_dir=None, force_download=False, force_extraction=False,
                 config_files=None):
        self.download_dir = os.path.expanduser(download_dir)
        self.extract_dir = os.path.expanduser(extract_dir or os.path.join(download_dir, "extracted"))
        self.force_download = force_download
        self.force_extraction = force_extraction
        self.config_files = config_files

        os.makedirs(self.download_dir, exist_ok=True)
        os.makedirs(self.extract_dir, exist_ok=True)

    def _extract_file(self, path, ignore_not_compressed=True):
        """
        Extract file.

        Parameters
        -----------
        path: str
            Compressed file path
        ignore_not_compressed: bool, optional
            Whether or not raise error if the file is not recognized as compressed file.

        Returns
        ---------
        str
            Extracted file path.
        """
        gz_extensions = (".gz",)
        zip_extensions = (".zip",)
        tar_extensions = (".tar", ".tar.bz2", ".tbz2", ".tbz", ".tb2", ".tar.gz")

        logger.info("Extracting {}".format(path))

        basename, extension = os.path.splitext(os.path.basename(path))

        if extension in zip_extensions:
            extract_func = extract_zip
        elif extension in tar_extensions:
            extract_func = extract_tar
        elif extension in gz_extensions:
            extract_func = extract_gz
        else:
            if ignore_not_compressed:
                logger.info("Extension {} not recognized as compressed file. Ignoring".format(extension))
                return path
            else:
                raise ValueError("Can't extract file {}. Supported extensions: ".format(
                                 path, ", ".join(zip_extensions + tar_extensions + gz_extensions)))

        extract_dir = os.path.join(self.extract_dir, basename)
        os.makedirs(extract_dir, exist_ok=True)

        return extract_func(path, extract_dir=extract_dir, force=self.force_extraction)

    def extract(self, path_or_paths, ignore_not_compressed=True):
        """
        Extract file/s.
        The currently supported file extensions are the following:

        - ``.gz``
        - ``.zip``
        - ``.tar``
        - ``.tar.bz2``
        - ``.tbz``
        - ``.tb2``
        - ``.tar.gz``

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
        if isinstance(path_or_paths, str):
            return self._extract_file(path_or_paths, ignore_not_compressed=ignore_not_compressed)
        elif isinstance(path_or_paths, dict):
            data_paths = {}
            for name, path in path_or_paths.items():
                data_paths[name] = self.extract(path, ignore_not_compressed=ignore_not_compressed)
            return data_paths
        elif isinstance(path_or_paths, Iterable):
            data_paths = []
            for path in path_or_paths:
                download_path = self.extract(path, ignore_not_compressed=ignore_not_compressed)
                data_paths.append(download_path)
            return data_paths
        else:
            raise ValueError("path_or_paths must be a str, an iterable or a dict. Got {}".format(type(path_or_paths)))

    def __getitem__(self, service_name):
        from gymnos.services import load

        return load(service_name, download_dir=self.download_dir,
                                  force_download=self.force_download,
                                  config_files=self.config_files)  # noqa: E127
