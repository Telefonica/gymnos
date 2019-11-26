#
#
#   SOFIA service
#
#

import os
import uuid
import time
import shutil
import logging
import requests

from urllib.parse import urlparse
from collections.abc import Iterable
from mimetypes import guess_extension

from .. import config

from .service import Service
from ..utils.text_utils import filenamify_url
from ..utils.downloader import download_file_from_url, urljoin


logger = logging.getLogger(__name__)


class SOFIA(Service):
    """
    Download components from SOFIA
    """

    SERVER_URL = "http://skywalker:8989"

    class Config(config.Config):
        """
        You need credentials to access SOFIA.

        Attributes
        ----------
        SOFIA_EMAIL: str
            Your SOFIA email
        SOFIA_PASSWORD: str
            Your password for your account
        """
        SOFIA_EMAIL = config.Value(required=True, help="SOFIA username")
        SOFIA_PASSWORD = config.Value(required=True, help="SOFIA password")

    def __init__(self, download_dir="downloads", force_download=False, config_files=None):
        super().__init__(download_dir=download_dir, force_download=force_download, config_files=config_files)

        self._auth_headers = None

    def _login(self):
        login_url = self.SERVER_URL + "/api/login"

        res = requests.post(login_url, data=dict(email=self.config.SOFIA_EMAIL,
                                                 password=self.config.SOFIA_PASSWORD))

        res.raise_for_status()

        res_json = res.json()

        self._auth_headers = {
            "Authorization": "Bearer {}".format(res_json["token"])
        }

        self._token_expiration = res_json["exp"]
        self._last_time_token_fetched = time.time()

    def _token_has_expired(self):
        elapsed_since_last_login = time.time() - self._last_time_token_fetched
        return elapsed_since_last_login > self._token_expiration

    def _login_if_needed(self):
        never_logged = self._auth_headers is None

        if never_logged or self._token_has_expired():
            self._login()

    def download(self, url_or_urls, verbose=True):
        """
        Download file/s from SOFIA

        Parameters
        ----------
        url_or_urls: str or list of str or dict(name: url)
            Url or urls to download
        verbose: bool, optional
            Whether or not show progress bar and logging info

        Returns
        -------
        str, list of str or dict
            File paths with downloaded SOFIA files.
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

    def _download_url(self, url, verbose=True):
        parsed = urlparse(url)

        assert parsed.scheme == "sofia"
        assert parsed.netloc in ("datasets", "models", "experiments")

        sofia_info_url = urljoin(self.SERVER_URL, "api", parsed.netloc, parsed.path)
        sofia_download_url = urljoin(sofia_info_url, "files")

        self._login_if_needed()

        res = requests.get(sofia_info_url, headers=self._auth_headers)
        res.raise_for_status()

        slug_url = filenamify_url(url)

        download_file_name = res.json().get("file", {}).get("name")

        if download_file_name is None:
            mime = res.json().get("file", {}).get("content_type", "")
            extension = guess_extension(mime) or ""
            filename = slug_url + extension
        else:
            filename = slug_url + "_" + download_file_name

        real_file_path = os.path.join(self.download_dir, filename)

        if os.path.isfile(real_file_path) and not self.force_download:
            if verbose:
                logger.info("Download for url {} found. Skipping".format(url))
            return real_file_path

        tmp_download_dir = os.path.join(self.download_dir, filename + ".tmp." + uuid.uuid4().hex)
        tmp_file_path = os.path.join(tmp_download_dir, filename)

        os.makedirs(tmp_download_dir)

        download_file_from_url(sofia_download_url, file_path=tmp_file_path, verbose=verbose,
                               force=self.force_download, headers=self._auth_headers)

        logger.info("Removing download temporary directory and moving files")
        shutil.move(tmp_file_path, real_file_path)
        shutil.rmtree(tmp_download_dir)
        return real_file_path
