#
#
#   Decompressor
#
#

import os
import zipfile
import gzip
import shutil

from ..logger import get_logger

logger = get_logger(prefix="Decompressor")


def decompress(file_path, delete_compressed=True):
    logger.info("Decompressing {}".format(file_path))

    if file_path.endswith(".zip"):
        decompress_zip(file_path)
    elif file_path.endswith(".gz"):
        decompress_gz(file_path)

    if not delete_compressed:
        return

    logger.info("Deleting compressed file {} ...".format(file_path))

    try:
        os.remove(file_path)
    except OSError as e:
        logger.warning("Could not delete compressed file, got %s" % e)


def decompress_zip(file_path):
    dir_path = os.path.dirname(file_path)

    try:
        with zipfile.ZipFile(file_path) as z:
            z.extractall(dir_path)
    except zipfile.BadZipfile as e:
        raise ValueError("Bad compressed file")


def decompress_gz(file_path):
    real_file_path = os.path.splitext(file_path)[0]

    with gzip.open(file_path, "r") as f_in, open(real_file_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def can_be_decompressed(filename):
    return (filename.endswith(".zip") or filename.endswith(".gz"))
