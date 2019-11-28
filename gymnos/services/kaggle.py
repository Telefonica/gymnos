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

from .. import config

from .service import Service
from ..utils.archiver import extract_zip
from ..utils.text_utils import print_table

logger = logging.getLogger(__name__)


class Kaggle(Service):
    """
    Download datasets/competitions from Kaggle.
    """

    class Config(config.Config):
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

        KAGGLE_USERNAME = config.Value(required=True, help="Kaggle username")
        KAGGLE_KEY = config.Value(required=True, help="Kaggle secret key")

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
        Download file/s from Kaggle.

        Parameters
        ----------
        dataset_name: str, optional
            Dataset name. Optional only if competition_name is None.
        competition_name: str, optional
            Competition name. Optional only if dataset_name is None.
        file_or_files: str or list of str or dict(name: url)
            File or files to download
        verbose: bool, optional
            Whether or not show progress bar and logging info

        Returns
        -------
        str, list of str or dict
            File paths with downloaded Kaggle files.
            The return type depends on the ``file_or_files`` parameter.
            If ``file_or_files`` is a str, it returns the file path.
            If ``file_or_files`` is a list of str, it returns a list with the file paths.
            If ``file_or_files`` is a dict, the return type is a dict(name: filepath)
        """
        if dataset_name is not None:
            dataset_or_competition_name = dataset_name
            file_downloader = self._download_dataset_file
            files_downloader = self._download_dataset_files
            dataset_files = self._kaggle_api.dataset_list_files(dataset_name)
        elif competition_name is not None:
            dataset_or_competition_name = competition_name
            file_downloader = self._download_competition_file
            files_downloader = self._download_competition_files
            dataset_files = self._kaggle_api.competition_list_files(competition_name)
        else:
            raise ValueError("You must set dataset_name or competition_name")

        dataset_files = getattr(dataset_files, "files", dataset_files)

        if dataset_files and verbose:
            if isinstance(file_or_files, str):
                to_include = [file_or_files]
            elif isinstance(file_or_files, dict):
                to_include = file_or_files.values()
            elif isinstance(file_or_files, (tuple, list, set)):
                to_include = file_or_files
            else:
                to_include = [file.ref for file in dataset_files]

            print_items = list(filter(lambda item: item.ref in to_include, dataset_files))
            print_table(print_items, ["name", "description", "size", "creationDate"], nrows=10)

        if file_or_files is None:
            return files_downloader(dataset_or_competition_name, verbose=verbose)
        if isinstance(file_or_files, str):
            return file_downloader(dataset_or_competition_name, filename=file_or_files, verbose=verbose)
        elif isinstance(file_or_files, dict):
            file_paths = {}
            for name, file in file_or_files.items():
                file_paths[name] = file_downloader(dataset_or_competition_name, file, verbose=verbose)
            return file_paths
        elif isinstance(file_or_files, (tuple, list, set)):
            file_paths = []
            for file in file_or_files:
                file_paths.append(file_downloader(dataset_or_competition_name, file, verbose=verbose))
            return file_paths
        else:
            raise ValueError("file_or_files must be a str, a dict or a list")

    def _download_file(self, kaggle_download_func, dataset_or_competition_name, filename, verbose=True):
        logger.info("Downloading {}:{} from Kaggle".format(dataset_or_competition_name, filename))

        resource_name = dataset_or_competition_name.replace("/", "_")
        real_file_path = os.path.join(self.download_dir, "kaggle_" + resource_name, filename)

        if os.path.isfile(real_file_path) and not self.force_download:
            logger.info("Download for {}/{} found. Skipping ...".format(dataset_or_competition_name, filename))
            return real_file_path

        tmp_download_dir = os.path.join(self.download_dir, "kaggle_" + resource_name + ".tmp." + uuid.uuid4().hex)

        kaggle_download_func(dataset_or_competition_name, file_name=filename,
                             path=tmp_download_dir, force=self.force_download,
                             quiet=not verbose)

        # some files are automatically compressed by Kaggle
        zip_file_path = os.path.join(tmp_download_dir, filename + ".zip")
        if os.path.isfile(zip_file_path):
            logger.info("Extracting zip file from download")
            extract_zip(zip_file_path, extract_dir=tmp_download_dir, force=self.force_download,
                        verbose=verbose)

        real_dirname, _ = os.path.splitext(real_file_path)
        os.makedirs(real_dirname, exist_ok=True)

        shutil.move(os.path.join(tmp_download_dir, filename), real_file_path)
        shutil.rmtree(tmp_download_dir)

        return real_file_path

    def _download_files(self, kaggle_download_func, dataset_or_competition_name, verbose=True):
        resource_name = dataset_or_competition_name.replace("/", "_")
        resource_download_dir = os.path.join(self.download_dir, "kaggle_" + resource_name)

        logger.info("Downloading {} from Kaggle".format(dataset_or_competition_name))

        kaggle_download_func(dataset_or_competition_name, path=resource_download_dir, force=self.force_download,
                             quiet=not verbose)

        if "/" in dataset_or_competition_name:
            owner_slug, dataset_slug = dataset_or_competition_name.split("/")
            zip_file_path = os.path.join(resource_download_dir, dataset_slug + ".zip")
            if os.path.isfile(zip_file_path):
                logger.info("Extracting zip file from download")
                extract_zip(zip_file_path, extract_dir=resource_download_dir, force=self.force_download,
                            verbose=verbose)

        return resource_download_dir

    def _download_dataset_file(self, dataset_name, filename, verbose=True):
        return self._download_file(self._kaggle_api.dataset_download_file, dataset_name, filename, verbose=verbose)

    def _download_competition_file(self, competition_name, filename, verbose=True):
        return self._download_file(self._kaggle_api.competition_download_file, competition_name, filename,
                                   verbose=verbose)

    def _download_dataset_files(self, dataset_name, verbose=True):
        return self._download_files(self._kaggle_api.dataset_download_files, dataset_name, verbose=verbose)

    def _download_competition_files(self, competition_name, verbose=True):
        return self._download_files(self._kaggle_api.competition_download_files, competition_name, verbose=verbose)
