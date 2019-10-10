#
#
#   Kaggle service
#
#

import os
import uuid
import shutil
import logging

from pydoc import locate
from collections import Iterable

from ..utils.archiver import extract_zip
from .service import Value, Service, ServiceConfig

logger = logging.getLogger(__name__)


class Kaggle(Service):
    """
    Download datasets/competitions from Kaggle.
    """

    class Config(ServiceConfig):
        """
        You need a Kaggle account to download from this service. If you don't have an account, sign up for a Kaggle account at
        http://www.kaggle.com/. Then go to the "Account" tab of your user profile (https://www.kaggle.com/<username>/account)
        and select "Create API Token". This will trigger the download of kaggle.json, a file containing your API credentials.

        Attributes
        ------------
        KAGGLE_USERNAME: str
            Kaggle username
        KAGGLE_KEY: str
            Kaggle API secret key
        """  # noqa: E501

        KAGGLE_USERNAME = Value(required=True, help="Kaggle username")
        KAGGLE_KEY = Value(required=True, help="Kaggle secret key")

    @property
    def _kaggle_api(self):
        """
        Get Kaggle api module and set username and key to environment variables.

        Returns
        --------
        kaggle.api
        """
        os.environ["KAGGLE_USERNAME"] = self.config.KAGGLE_USERNAME
        os.environ["KAGGLE_KEY"] = self.config.KAGGLE_KEY

        return locate("kaggle.api")

    def download(self, dataset_name=None, competition_name=None, file_or_files=None, verbose=True):
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
            file_or_files = self.list_files(dataset_name=dataset_name, competition_name=competition_name)

        if isinstance(file_or_files, str):
            if dataset_name is not None:
                logger.info("Downloading {} from Kaggle dataset {}".format(file_or_files, dataset_name))
                resource_name = dataset_name.replace("/", "_")
            elif competition_name is not None:
                logger.info("Downloading {} from Kaggle competition {}".format(file_or_files, competition_name))
                resource_name = competition_name
            else:
                raise ValueError("You must specify dataset_name or competition_name")

            real_file_path = os.path.join(self.download_dir, resource_name + "_" + file_or_files)

            if os.path.isfile(real_file_path) and not self.force_download:
                if verbose:
                    logger.debug("Download for {}/{} found. Skipping".format(resource_name, file_or_files))
                return real_file_path

            tmp_download_dir = os.path.join(self.download_dir, "kaggle" + "_" + resource_name + "_" + file_or_files +
                                            ".tmp." + uuid.uuid4().hex)

            self._download(dataset_name=dataset_name, competition_name=competition_name, files=[file_or_files],
                           download_dir=tmp_download_dir)

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

    def _download(self, dataset_name=None, competition_name=None, files=None, download_dir=".", verbose=True):
        """
        Download dataset or competition from Kaggle

        Parameters
        -----------
        dataset_name: str
            Kaggle dataset name with the format <user>/<dataset_name>, e.g mlg-ulb/creditcardfraud
        competition_name: str
            Kaggle competition name, e.g titanic
        files: list of str, optional
            Specific files to download
        download_dir: str, optional
            Directory to download files. By default, current directory
        verbose: bool, optional
            Whether or not show progress bar
        """
        if dataset_name is not None:
            self._download_dataset(dataset_name, files=files, download_dir=download_dir,
                                   verbose=verbose)
        elif competition_name is not None:
            self._download_competition(competition_name, files=files, download_dir=download_dir,
                                       verbose=verbose)
        else:
            raise ValueError("You must specify dataset_name or competition_name")

    def _download_dataset(self, name, files=None, download_dir=".", verbose=True):
        """
        Download dataset from Kaggle.

        Parameters
        -----------
        name: str, optional
            Kaggle dataset name with the format <user>/<dataset_name>, e.g mlg-ulb/creditcardfraud
        files: list of str, optional
            Specific files to download.
        download_dir: str, optional
            Directory to download files. By default, current directory
        verbose: bool, optional
            Whether or not show progress bar.
        """
        if files is None:
            self._kaggle_api.dataset_download_files(name, path=download_dir, force=self.force_download, unzip=True,
                                                    quiet=not verbose)
        elif isinstance(files, (list, set, tuple)):
            for filename in files:
                self._kaggle_api.dataset_download_file(name, file_name=filename, path=download_dir,
                                                       force=self.force_download, quiet=not verbose)
                zip_file_path = os.path.join(download_dir, filename + ".zip")
                if os.path.isfile(zip_file_path):
                    extract_zip(zip_file_path, extract_dir=download_dir, force=self.force_download,
                                verbose=verbose)
        else:
            raise ValueError("files must be None or a list")

    def _download_competition(self, name, files=None, download_dir=".", verbose=True):
        """
        Download competition from Kaggle.

        Parameters
        -----------
            name: str, optional
                Kaggle competition name, e.g titanic
            files: list of str, optional
                Specific files to download
            download_dir: str, optional
                Directory to download files. By default, current directory
            verbose: bool, optional
                Whether or not show progress bar.
        """
        if files is None:
            self._kaggle_api.competition_download_files(name, path=download_dir, force=self.force_download,
                                                        quiet=not verbose)
        elif isinstance(files, (list, set, tuple)):
            for filename in files:
                self._kaggle_api.competition_download_file(name, filename, download_dir,
                                                           force=self.force_download, quiet=not verbose)
        else:
            raise ValueError("files must be None or a list")

    def list_competition_files(self, competition_name):
        """
        List competition files
        Parameters
        ----------
        competition_name: str
            Competition name
        """
        files = self._kaggle_api.competition_list_files(competition_name)
        if files:
            return [file.ref for file in files]

    def list_dataset_files(self, dataset_name):
        """
        List dataset files
        Parameters
        ----------
        dataset_name: str, optional
            Kaggle dataset name with the format <user>/<dataset_name>, e.g mlg-ulb/creditcardfraud
        Returns
        --------
        list of str
            List of filenames
        """
        result = self._kaggle_api.dataset_list_files(dataset_name)
        if result:
            return [file.ref for file in result.files]

    def list_files(self, dataset_name=None, competition_name=None):
        """
        List dataset or competition files

        Parameters
        -----------
        dataset_name: str, optional
            Kaggle dataset name with the format <user>/<dataset_name>, e.g mlg-ulb/creditcardfraud.
            Mandatory if ``competition_name`` is None.
        competition_name: str, optional
            Kaggle competition name.
            Mandatory if ``dataset_name`` is None.
        """
        if dataset_name is not None:
            return self._kaggle_api.list_dataset_files(dataset_name)
        elif competition_name is not None:
            return self._kaggle_api.list_competition_files(competition_name)
        else:
            raise ValueError("You must specify dataset_name or competition_name")
