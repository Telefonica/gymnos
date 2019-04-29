#
#
# Kaggle Mixin
#
#

from pydoc import locate

from ...logger import get_logger
from ...services.kaggle_dataset_downloader import KaggleDatasetDownloader


class KaggleMixin:
    """
    Mixin to download Kaggle datasets. It provides implementation for ``download`` method.

    Attributes
    ----------
    kaggle_dataset_name: str
        Kaggle dataset or competition to download in the format `username/dataset`.
    kaggle_dataset_files: str or list of str, optional
        Specific files to download (by default whole dataset).
    """

    kaggle_dataset_name = None
    kaggle_dataset_files = None


    def download(self, download_path):
        """
        Download Kaggle dataset/competition specified by ``kaggle_dataset_name``
        and ``kaggle_dataset_files``.

        Parameters
        ----------
        download_path: str
            Path to download kaggle dataset.

        Note
        ----
        You need to provide Kaggle credentials to download a dataset from Kaggle.
        Sign up for a Kaggle account at https://www.kaggle.com. Then go to the 'Account' tab of
        your user profile (``https://www.kaggle.com/<username>/account``) and select 'Create API Token'.
        This will trigger the download of ``kaggle.json``, a file containing your API credentials.
        Place this file in the location ``~/.kaggle/kaggle.json`` (on Windows in the location
        ``C:\\Users\\<Windows-username>\\.kaggle\\kaggle.json`` - you can check the exact location, sans drive,
        with ``echo %HOMEPATH%``).
        You can define a shell environment variable ``KAGGLE_CONFIG_DIR`` to change this location to
        ``$KAGGLE_CONFIG_DIR/kaggle.json`` (on Windows it will be ``%KAGGLE_CONFIG_DIR%\\kaggle.json``).
        For your security, ensure that other users of your computer do not have read access to your credentials.
        On Unix-based systems you can do this with the following command:

        .. code-block:: sh

            chmod 600 ~/.kaggle/kaggle.json

        You can also choose to export your Kaggle username and token to the environment:

        .. code-block:: sh

            export KAGGLE_USERNAME=datadinosaur
            export KAGGLE_KEY=xxxxxxxxxxxxxx

        In addition, you can export any other configuration value that normally would be in the
        ``$HOME/.kaggle/kaggle.json`` in the format ``KAGGLE_`` (note uppercase).
        """

        if self.kaggle_dataset_name is None:
            raise ValueError("kaggle_dataset_name cannot be None")

        logger = get_logger(prefix=self)
        downloader = KaggleDatasetDownloader()
        kaggle_api_exception = locate("kaggle.rest.ApiException")

        try:
            downloader.download(self.kaggle_dataset_name, self.kaggle_dataset_files,
                                download_path, verbose=True)
        except kaggle_api_exception:
            logger.error("Error downloading Kaggle dataset. Check your credentials.")
            raise
