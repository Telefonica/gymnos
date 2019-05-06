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
    """
    Manage HDF5 files (save and read).

    Parameters
    -----------
    file_path: str
        HDF5 file path.
    """

    def __init__(self, file_path):
        self.file_path = file_path

        self.logger = get_logger(prefix=self)

    def exists(self):
        """
        HDF5 file given by ``file_path`` already exists.

        Returns
        -------
        bool
        """
        return os.path.isfile(self.file_path)

    def retrieve(self, key):
        """
        Retrieve Pandas or Numpy array from HDF5 file.

        Parameters
        ----------
        key: str
            Key to retrieve

        Returns
        -------
        data: array_like
            Any data saved for ``key`` parameter.
        """
        try:
            dataset = self.retrieve_pandas(key)
            self.logger.debug("Retrieved Pandas dataset from HDF5 {} key ({})".format(key, self.file_path))
        except TypeError:
            dataset = self.retrieve_numpy(key)
            self.logger.debug("Retrieved Numpy dataset from HDF5 {} key ({})".format(key, self.file_path))
        return dataset

    def retrieve_pandas(self, key):
        """
        Retrieve Pandas array from HDF5 file. Array is fully loaded in memory.

        Parameters
        ----------
        key: str
            Key to retrieve

        Returns
        -------
        data: pd.Series or pd.DataFrame
        """
        return pd.read_hdf(self.file_path, key)

    def retrieve_numpy(self, key):
        """
        Retrieve NumPy array from HDF5 file. Array is fully loaded in memory.

        Parameters
        ----------
        key: str
            Key to retrieve

        Returns
        -------
        data: np.array
        """
        with h5py.File(self.file_path) as h5f:
            return h5f[key][:]

    def save(self, key, data):
        """
        Save Pandas or NumPy array to HDF5 file.

        Parameters
        ----------
        key: str
            Key to save data inside HDF5 file.
        data: pd.Series or pd.DataFrame or np.array
            Data to save.
        """
        if isinstance(data, (pd.Series, pd.DataFrame)):
            self.logger.debug("Saving Pandas dataset to HDF5 {} key ({})".format(key, self.file_path))
            self.save_pandas(key, data)
        else:
            self.logger.debug("Saving Numpy dataset to HDF5 {} key ({})".format(key, self.file_path))
            self.save_numpy(key, data)

    def save_numpy(self, key, data):
        """
        Save NumPy array to HDF5 file.

        Parameters
        ----------
        key: str
            Key to save data inside HDF5 file.
        data: np.array
            Data to save.
        """
        with h5py.File(self.file_path) as h5f:
            dst = h5f.create_dataset(key, shape=data.shape, dtype=data.dtype, compression="gzip")
            chunk_size = len(data) // 10
            for idx in trange(0, len(data), chunk_size):
                dst[idx:idx + chunk_size] = data[idx:idx + chunk_size]

    def save_pandas(self, key, data):
        """
        Save Pandas Series or DataFrame to HDFD5 file.

        Parameters
        ----------
        key: str
            Key to save data inside HDF5 file.
        data: pd.Series or pd.DataFrame
            Data to save.
        """
        data.to_hdf(self.file_path, key, complib='zlib', complevel=5, format="table")
