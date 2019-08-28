#
#
#   SAMBA service
#
#

import os
import uuid
import shutil
import logging

from urllib.parse import urlparse
from collections import Iterable

from ..utils.hashing import sha1_text
from ..utils.text_utils import filenamify_url
from ..utils.downloader import download_file_from_smb
from .service import Service, ServiceConfig, Value

logger = logging.getLogger(__name__)


class SMB(Service):
    """
    Download files from SAMBA servers.
    """

    class Config(ServiceConfig):
        """
        You need credentials to access SAMBA servers.

        Attributes
        -------------
        SMB_USERNAME: str
            Your username for SAMBA server
        SMB_PASSWORD: str
            Your password for SAMBA server
        """

        SMB_USERNAME = Value(required=True, help="SAMBA server username")
        SMB_PASSWORD = Value(required=True, help="SAMBA server password")

    def _download_url(self, url, verbose=True):
        """
        Download file from SAMBA server.

        Parameters
        ------------
        url: str
            Url to download
        verbose: bool, optional
            Whether or not show progress bar and logging info

        Returns
        ---------
        file_path: str
            Downloaded file path
        """
        logger.info("Downloading file from SAMBA uri {}".format(url))

        if urlparse(url).scheme != "smb":
            raise ValueError(("Url {} is not a SAMBA uri. It must have the following format: "
                              "smb://<ip>:[<port>]/<drive>/<path>").format(url))

        sha1_url_hash = sha1_text(url)
        slug_url = filenamify_url(url)

        if len(slug_url) > 90:
            slug_url = slug_url[:45] + "_" + slug_url[-45:]

        filename = sha1_url_hash + "_" + slug_url

        real_file_path = os.path.join(self.download_dir, filename)

        if os.path.isfile(real_file_path) and not self.force_download:
            if verbose:
                logger.info("Download for SAMBA uri {} found. Skipping".format(url))
            return real_file_path

        tmp_download_dir = os.path.join(self.download_dir, filename + ".tmp." + uuid.uuid4().hex)
        tmp_file_path = os.path.join(tmp_download_dir, filename)

        os.makedirs(tmp_download_dir)

        download_file_from_smb(url, file_path=tmp_file_path, username=self.config.SMB_USERNAME,
                               password=self.config.SMB_PASSWORD, verbose=verbose, force=self.force_download)

        logger.info("Removing download temporary directory and moving files")
        shutil.move(tmp_file_path, real_file_path)
        shutil.rmtree(tmp_download_dir)
        return real_file_path

    def download(self, url_or_urls, verbose=True):
        """
        Download file/s from SAMBA server

        Parameters
        ----------
        url_or_urls: str or list of str or dict(name: url)
            Url or urls to download
        verbose: bool, optional
            Whether or not show progress bar and logging info

        Returns
        -------
        str, list of str or dict
            File paths with downloaded SAMBA files.
            The return type depends on the ``url_or_urls`` parameter.
            If ``url_or_urls`` is a str, it returns the file path.
            If ``url_or_urls`` is a list of str, it returns a list with the file paths.
            If ``url_or_urls`` is a dict, the return type is a dict(name: filepath)
        """
        if isinstance(url_or_urls, str):
            return self._download_url(url_or_urls, verbose=verbose)
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
