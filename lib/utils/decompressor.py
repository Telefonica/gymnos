#
#
#   Decompressor
#
#

import os
import zipfile
import gzip
import shutil

from ..logger import logger


class Decompressor:

    @staticmethod
    def extract(file_path, delete_compressed=True):
        logger.info("Decompressing {} ...".format(file_path))

        if file_path.endswith(".zip"):
            Decompressor.extract_zip(file_path)
        elif file_path.endswith(".gz"):
            Decompressor.extract_gz(file_path)

        if not delete_compressed:
            return

        logger.info("Deleting compressed file {} ...".format(file_path))

        try:
            os.remove(file_path)
        except OSError as e:
            logger.warning("Could not delete compressed file, got %s" % e)


    @staticmethod
    def extract_zip(file_path):
        dir_path = os.path.dirname(file_path)

        try:
            with zipfile.ZipFile(file_path) as z:
                z.extractall(dir_path)
        except zipfile.BadZipfile as e:
            raise ValueError("Bad compressed file")


    @staticmethod
    def extract_gz(file_path):
        real_file_path = os.path.splitext(file_path)[0]

        with gzip.open(file_path, "r") as f_in, open(real_file_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    @staticmethod
    def is_compressed(filename):
        return (filename.endswith(".zip") or filename.endswith(".gz"))
