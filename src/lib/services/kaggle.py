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
        files = KaggleService.get_kaggle().competition_list_files(competition_name)
        if files:
            return [file.ref for file in files]

    @staticmethod
    def list_dataset_files(dataset_name):
        result = KaggleService.get_kaggle().dataset_list_files(dataset_name)
        if result:
            return [file.ref for file in result.files]

    @staticmethod
    def list_files(dataset_name=None, competition_name=None):
        if dataset_name is not None:
            return KaggleService.list_dataset_files(dataset_name)
        elif competition_name is not None:
            return KaggleService.list_competition_files(competition_name)
        else:
            raise ValueError("You must specify dataset_name or competition_name")
