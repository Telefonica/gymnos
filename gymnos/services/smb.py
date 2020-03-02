#
#
#   SAMBA service
#
#

import os
import uuid
import shutil
import socket
import logging
import pathlib

from urllib.error import URLError
from urllib.parse import urlparse
from collections.abc import Iterable
from contextlib import contextmanager

from .. import config

from .service import Service
from ..utils.hashing import sha1_text
from ..utils.text_utils import filenamify_url
from ..utils.lazy_imports import lazy_imports as lazy


logger = logging.getLogger(__name__)


def parse_smb_uri(smb_uri):
    uri_parsed = urlparse(smb_uri)

    smb_server_port = uri_parsed.port or 445
    smb_server_ip = uri_parsed.hostname

    smb_file_path = pathlib.Path(uri_parsed.path)

    shared_drive, *smb_file_path = smb_file_path.parts[1:]
    smb_file_path = os.path.join(*smb_file_path)

    return dict(
        ip=smb_server_ip,
        port=smb_server_port,
        drive=shared_drive,
        path=smb_file_path
    )


@contextmanager
def smb_connection(ip, port=445, username=None, password=None):
    server_name = socket.gethostbyaddr(ip)
    if server_name:
        server_name = server_name[0]
    else:
        raise URLError('Hostname with ip {} does not reply back with its machine name'.format(ip))

    client_name = socket.gethostname()
    if client_name:
        client_name = client_name.split('.')[0]
    else:
        client_name = 'SMB%d' % os.getpid()

    smb = __import__("{}.SMBConnection".format(lazy.smb.__name__))

    with smb.SMBConnection.SMBConnection(username, password, client_name, server_name, use_ntlm_v2=True,
                                         is_direct_tcp=True) as conn:
        success = conn.connect(ip, port)
        if not success:
            raise ValueError("Authentication to {} failed. Check your credentials".format(ip))

        yield conn


class SMB(Service):
    """
    Download files from SAMBA servers.
    """

    class Config(config.Config):
        """
        Some SAMBA servers require credentials. If credentials not provided, connection will be as ``guest``.

        Attributes
        -------------
        SMB_USERNAME: str, optional
            Your username for SAMBA server
        SMB_PASSWORD: str, optional
            Your password for SAMBA server
        """

        SMB_USERNAME = config.Value(required=False, default="guest", help="SAMBA server username")
        SMB_PASSWORD = config.Value(required=False, default="", help="SAMBA server password")

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

        uri_components = parse_smb_uri(url)

        sha1_url_hash = sha1_text(url)
        slug_url = filenamify_url(url)

        if len(slug_url) > 90:
            slug_url = slug_url[:45] + "_" + slug_url[-45:]

        filename = sha1_url_hash + "_" + slug_url

        real_file_path = os.path.join(self.download_dir, filename)

        if self.config.SMB_USERNAME == "guest":
            logger.warning("You're connecting as guest. Credentials may be required")

        smb_conn_params = dict(username=self.config.SMB_USERNAME,
                               password=self.config.SMB_PASSWORD,
                               ip=uri_components["ip"],
                               port=uri_components["port"])

        with smb_connection(**smb_conn_params) as conn:
            if os.path.isfile(real_file_path) and not self.force_download:
                if verbose:
                    logger.info("Download for SAMBA uri {} found. Skipping".format(url))
                return real_file_path

            tmp_download_dir = os.path.join(self.download_dir, filename + ".tmp." + uuid.uuid4().hex)
            tmp_file_path = os.path.join(tmp_download_dir, filename)

            os.makedirs(tmp_download_dir)

            with open(tmp_file_path, "wb") as fp:
                conn.retrieveFile(uri_components["drive"], uri_components["path"], fp)

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
