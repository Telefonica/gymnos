#
#
#   HDF Manager
#
#

import os
import h5py
import pandas as pd

from ..logger import get_logger

from tqdm import trange


class HDFManager:

    def __init__(self, file_path):
        self.file_path = file_path

        self.logger = get_logger(prefix=self)

    def exists(self):
        return os.path.isfile(self.file_path)

    def retrieve(self, key):
        try:
            dataset = self.retrieve_pandas(key)
            self.logger.info("Retrieving Pandas dataset from HDF5 ({})".format(self.file_path))
        except TypeError:
            dataset = self.retrieve_numpy(key)
            self.logger.info("Retrieving Numpy dataset from HDF5 ({})".format(self.file_path))
        return dataset

    def retrieve_pandas(self, key):
        return pd.read_hdf(self.file_path, key)

    def retrieve_numpy(self, key):
        with h5py.File(self.file_path) as h5f:
            return h5f[key][:]

    def save(self, key, data):
        if isinstance(data, (pd.Series, pd.DataFrame)):
            self.logger.info("Saving Pandas dataset to HDF5 ({})".format(self.file_path))
            self.save_pandas(key, data)
        else:
            self.logger.info("Saving Numpy dataset to HDF5 ({})".format(self.file_path))
            self.save_numpy(key, data)

    def save_numpy(self, key, data):
        with h5py.File(self.file_path) as h5f:
            dst = h5f.create_dataset(key, shape=data.shape, dtype=data.dtype, compression="gzip")
            chunk_size = len(data) // 10
            for idx in trange(0, len(data), chunk_size):
                dst[idx:idx + chunk_size] = data[idx:idx + chunk_size]

    def save_pandas(self, key, data):
        data.to_hdf(self.file_path, key, complib='zlib', complevel=5, format="table")
