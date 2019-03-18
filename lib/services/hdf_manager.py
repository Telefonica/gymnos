#
#
#   HDF Manager
#
#

import os
import h5py
import pandas as pd


class HDFManager:

    def __init__(self, file_path):
        self.file_path = file_path

    def exists(self):
        return os.path.isfile(self.file_path)

    def retrieve(self, key):
        try:
            return self.retrieve_pandas(key)
        except TypeError:
            return self.retrieve_numpy(key)

    def retrieve_pandas(self, key):
        return pd.read_hdf(self.file_path, key)

    def retrieve_numpy(self, key):
        with h5py.File(self.file_path) as h5f:
            return h5f[key][:]

    def save(self, key, data):
        if isinstance(data, (pd.Series, pd.DataFrame)):
            self.save_pandas(key, data)
        else:
            self.save_numpy(key, data)

    def save_numpy(self, key, data):
        with h5py.File(self.file_path) as h5f:
            h5f.create_dataset(key, data=data, shape=data.shape, compression="gzip", compression_opts=9)

    def save_pandas(self, key, data):
        data.to_hdf(self.file_path, key, complib='zlib', complevel=5, format="table")
