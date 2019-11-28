#
#
#   Service
#
#

import logging

from .. import config

logger = logging.getLogger(__name__)


class Service:
    """
    Base class for all services.
    """

    class Config(config.Config):
        """
        Base config for all services.

        Parameters
        -----------
        download_dir: str, optional
            Directory to download files
        force_download: bool, optional
            Whether or not force download if file exists
        config_files: list of str
            Files to search for configuration variables.
        """

    def __init__(self, download_dir="downloads", force_download=False, config_files=None):
        self.download_dir = download_dir
        self.force_download = force_download

        self.config = self.Config(files=config_files)
        self.config.load()
