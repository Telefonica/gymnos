#
#
#   Decompressor
#
#

import os
import gzip
import shutil
import tarfile
import zipfile

from tqdm import tqdm

from ..logger import get_logger

logger = get_logger(prefix="decompressor")


def extract_zip(file_path, extract_dir=".", force=False, verbose=True):
    """
    Extract .zip file to directory

    Parameters
    ----------
    file_path: str
        Zipped file path.
    extract_dir: str, optional
        Directory to extract contents. By default, current directory
    force: bool
        Whether or not force extraction. By default, if the extracted file exists, don't extract.
    verbose: bool
        Whether or not show progress bar

    Returns
    -------
    extract_dir
        Directory where files are extracted
    """
    with zipfile.ZipFile(file_path) as zip_file:
        iterator = zip_file.namelist()
        if verbose:
            iterator = tqdm(iterable=zip_file.namelist(), total=len(zip_file.namelist()))
        for file in iterator:
            file_path = os.path.join(extract_dir, file)
            if os.path.isfile(file_path) and not force:
                continue
            zip_file.extract(member=file, path=extract_dir)

    return extract_dir


def extract_gz(file_path, extract_dir=".", force=False):
    """
    Extract .gz file to directory.

    Parameters
    ----------
    file_path: str
        Gz file path.
    extract_dir: str, optional
        Directory to extract file. *.gz files can only contain one file. By default, current directory
    force: bool, optional
        Whether or not force extraction if file exists

    Returns
    -------
    extracted_file_path: str
        Path where file is extracted
    """
    filename = os.path.splitext(os.path.basename(file_path))[0]
    extracted_file_path = os.path.join(extract_dir, filename)

    if os.path.isfile(extracted_file_path) and not force:
        print("Extracted file for {} found. Skipping".format(filename))
        return extracted_file_path

    with gzip.open(file_path, "r") as f_in, open(extracted_file_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    return extracted_file_path


def extract_tar(file_path, extract_dir=".", force=False, verbose=True):
    """
    Extract .tar file

    Parameters
    ----------
    file_path: str
        TAR file path
    extract_dir: bool, optiona
        Directory to extract file. By default, current directory
    force: bool, optional
        Whether or not force extraction if file exists.
    verbose: bool, optional
        Whether or not show progress bar

    Returns
    -------
    extract_dir
        Directory where files are extracted
    """
    with tarfile.open(file_path) as tarball:
        iterator = tarball.getmembers()
        if verbose:
            iterator = tqdm(iterable=iterator, total=len(tarball.getmembers()))
        for member in iterator:
            file_path = os.path.join(extract_dir, member)
            if os.path.isfile(file_path) and not force:
                continue
            tarball.extract(member=member)

    return extract_dir
