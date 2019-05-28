#
#
#   Kaggle
#
#

import os

from pydoc import locate
from collections import Iterable

from ..utils.extractor import extract_zip


class KaggleService:

    @staticmethod
    def get_kaggle():
        return locate("kaggle.api")

    @staticmethod
    def download(dataset_name=None, competition_name=None, files=None, download_dir=".", force=False, verbose=True):
        """
        Download dataset or competition from Kaggle.

        Parameters
        -----------
            dataset_name: str, optional
                Kaggle dataset name with the format <user>/<dataset_name>, e.g mlg-ulb/creditcardfraud
                Mandatory if ``competition_name`` is None.
            competition_name: str, optional
                Kaggle competition name, e.g titanic.
                Mandatory if ``dataset_name`` is None.
            files: list of str, optional
                Specific files to download
            download_dir: str, optional
                Directory to download files. By default, current directory
            force: bool, optional
                Whether or not force download if file already exists.
            verbose: bool, optional
                Whether or not show progress bar.
        """
        if dataset_name is not None:
            KaggleService.download_dataset(dataset_name=dataset_name, files=files, download_dir=download_dir,
                                           force=force, verbose=verbose)
        elif competition_name is not None:
            KaggleService.download_competition(competition_name=competition_name, files=files,
                                               download_dir=download_dir, force=force, verbose=verbose)
        else:
            raise ValueError("You must specify dataset_name or competition_name")

    @staticmethod
    def download_dataset(dataset_name, files=None, download_dir=".", force=False, verbose=True):
        """
        Download dataset from kaggle.

        Parameters
        -----------
            dataset_name: str, optional
                Kaggle dataset name with the format <user>/<dataset_name>, e.g mlg-ulb/creditcardfraud
            files: list of str, optional
                Specific files to download.
            download_dir: str, optional
                Directory to download files. By default, current directory
            force: bool, optional
                Whether or not force download if file already exists.
            verbose: bool, optional
                Whether or not show progress bar.
        """
        if files is None:
            KaggleService.get_kaggle().dataset_download_files(dataset_name, path=download_dir, force=force, unzip=True,
                                                              quiet=not verbose)
        elif isinstance(files, Iterable):
            for filename in files:
                KaggleService.get_kaggle().dataset_download_file(dataset_name, file_name=filename, path=download_dir,
                                                                 force=force, quiet=not verbose)
                zip_file_path = os.path.join(download_dir, filename + ".zip")
                if os.path.isfile(zip_file_path):
                    extract_zip(zip_file_path, extract_dir=download_dir, force=force, verbose=verbose)
        else:
            raise ValueError("files must be None or a list")

    @staticmethod
    def download_competition(competition_name, files=None, download_dir=".", force=False, verbose=True):
        """
        Download competition from Kaggle.

        Parameters
        -----------
            competition_name: str, optional
                Kaggle competition name, e.g titanic
            files: list of str, optional
                Specific files to download
            download_dir: str, optional
                Directory to download files. By default, current directory
            force: bool, optional
                Whether or not force download if file already exists.
            verbose: bool, optional
                Whether or not show progress bar.
        """
        if files is None:
            KaggleService.get_kaggle().competition_download_files(competition_name, path=download_dir, force=force,
                                                                  quiet=not verbose)
        elif isinstance(files, Iterable):
            for filename in files:
                KaggleService.get_kaggle().competition_download_file(competition_name, filename, download_dir,
                                                                     force=force, quiet=not verbose)
        else:
            raise ValueError("files must be None or a list")

    @staticmethod
    def list_competition_files(competition_name):
        """
        List competition files

        Parameters
        ----------
        competition_name: str
            Competition name
        """
        files = KaggleService.get_kaggle().competition_list_files(competition_name)
        if files:
            return [file.ref for file in files]

    @staticmethod
    def list_dataset_files(dataset_name):
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
        result = KaggleService.get_kaggle().dataset_list_files(dataset_name)
        if result:
            return [file.ref for file in result.files]

    @staticmethod
    def list_files(dataset_name=None, competition_name=None):
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
            return KaggleService.list_dataset_files(dataset_name)
        elif competition_name is not None:
            return KaggleService.list_competition_files(competition_name)
        else:
            raise ValueError("You must specify dataset_name or competition_name")
