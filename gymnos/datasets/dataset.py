#
#
#   Dataset
#
#

import pickle
import numpy as np

from tqdm import tqdm
from functools import lru_cache
from abc import ABCMeta, abstractmethod

from ..utils.io_utils import read_lines
from ..utils.data import IterableDataLoader, collate


class BaseDataset(metaclass=ABCMeta):
    """
    Common methods to Gymnos datasets.

    All Gymnos datasets have the following properties:
        - ``features_info``: information about features
        - ``labels_info``: information about labels
    All Gymnos datasets have the following methods:
        - ``download_and_prepare``: download files associated with dataset.
        - ``load``: load full dataset
        - ``to_hdf5``: convert dataset to HDF5
        - ``__iter__``: iterate over rows of dataset
        - ``__len__``: get number of rows of dataset
    """

    @property
    @abstractmethod
    def features_info(self):
        """
        Returns information about your features.
        """

    @property
    @abstractmethod
    def labels_info(self):
        """
        Returns information about your labels.

        Returns
        --------
        Array
        """

    @abstractmethod
    def load(self):
        """
        Load dataset dataset into memory
        """

    @abstractmethod
    def __iter__(self):
        """
        Iterate over rows of dataset
        """

    @abstractmethod
    def __len__(self):
        """
        Returns length of dataset
        """

    def download_and_prepare(self, dl_manager):
        """
        Optional method to download external data if needed.

        Parameters
        ------------
        dl_manager: gymnos.services.DownloadManager
            Download Manager
        """

    def to_hdf5(self, file_path, features_key="features", labels_key="labels", info_key="info", chunk_size=None,
                compression="gzip", compression_opts=None, force=False):
        """
        Export dataset to HDF5 file

        Parameters
        ----------
        file_path: str
            File path to export dataset
        features_key: str, optional
            HDF5 key to save features
        labels_key: str, optional
            HDF5 key to save labels
        info_key: str, optional
            HDF5 key to save info
        chunk_size: int, optional
            Chunk size to read dataset. By default, full dataset is read.
        compression: str, optional
            HDF5 compression algorithm
        compression_opts: dict, optional
            HDF5 compression options, check h5py documentation to see more.
        force: bool, optional
            Whether or not overwrite file if it already exists
        """
        import h5py

        if chunk_size is None:
            chunk_size = len(self)

        # sequence is also an iterable so to support both datasets we load samples with
        # IterableDataLoader
        data_loader = IterableDataLoader(self, batch_size=chunk_size)

        mode = "w" if force else "x"

        with h5py.File(file_path, mode) as h5f:
            labels_shape = [len(self)] + self.labels_info.shape
            features_shape = [len(self)] + self.features_info.shape

            labels_dtype = self.labels_info.dtype
            features_dtype = self.features_info.dtype

            if features_dtype == str:
                features_dtype = h5py.special_dtype(vlen=str)

            features = h5f.create_dataset(features_key, shape=features_shape, dtype=features_dtype,
                                          compression=compression, compression_opts=compression_opts)
            labels = h5f.create_dataset(labels_key, shape=labels_shape, compression=compression,
                                        dtype=labels_dtype, compression_opts=compression_opts)

            features.attrs[info_key] = np.string_(pickle.dumps(self.features_info))
            labels.attrs[info_key] = np.string_(pickle.dumps(self.labels_info))

            for index, (X, y) in enumerate(tqdm(data_loader)):
                start = index * data_loader.batch_size
                end = start + data_loader.batch_size
                features[start:end] = X
                labels[start:end] = y


class Dataset(BaseDataset):
    """
    Base methods that all datasets must implement.
    This will be the common API all datasets will support.
    """

    @abstractmethod
    def __getitem__(self, index):
        """
        Get row by index

        Parameters
        ----------
        index: int
            Index

        Returns
        --------
        features: array-like
            Features for row
        labels: array-like
            Labels for row
        """

    @abstractmethod
    def __len__(self, index):
        """
        Returns number of rows
        """

    def __iter__(self):
        """
        Iterate over rows.

        Yields
        ------
        features: array-like
            Features for yielded row
        labels: array-like
            Labels for yielded row
        """
        for item in (self[i] for i in range(len(self))):
            yield item

    def load(self):
        """
        Load features and labels.

        Returns
        --------
        features: array-like
            Features
        labels: array-like
            Labels
        """
        X, y = zip(*self)
        return collate(X), collate(y)


class IterableDataset(BaseDataset):
    """
    Iterable dataset
    """

    @abstractmethod
    def __iter__(self):
        """
        Iterate over rows.

        Yields
        ------
        features: array-like
            Features for yielded row
        labels: array-like
            Labels for yielded row
        """

    @abstractmethod
    def __len__(self):
        """
        Returns number of rows.
        """

    def load(self):
        """
        Load dataset into memory

        Returns
        --------
        features: array-like
            Dataset features
        labels: array-like
            Dataset labels
        """
        X, y = zip(*self)
        return collate(X), collate(y)


class SparkDataset(BaseDataset):
    """
    Dataset for Spark DataFrames.

    Parameters
    ------------
    features_col: str
        Name of the column with the features
    labels_col: str
        Name of the column with the labels
    """

    def __init__(self, features_col="features", labels_col="labels"):
        self.features_col = features_col
        self.labels_col = labels_col

    @abstractmethod
    def load(self):
        """
        Load Spark DataFrame

        Returns
        --------
        df: pyspark.sql.DataFrame
            DataFrame
        """

    @property
    def spark(self):
        """
        Returns default Spark session
        """
        from pyspark.sql import SparkSession
        return SparkSession.builder.getOrCreate()

    def __iter__(self):
        """
        Iterate over rows

        Yields
        -------
        features: array-like
            Features for yielded row
        labels: array-like
            Labels for yielded row
        """
        df = self.load()
        for row in df.select(self.features_col, self.labels_col).rdd.toLocalIterator():
            yield row[self.features_col], row[self.labels_col]

    @lru_cache()  # dataset length never changes so we can cache it
    def __len__(self):
        """
        Returns number of rows.
        """
        return self.load().count()


class Array:
    """
    Info about data shape and data type.

    Parameters
    ----------
    shape: list
        Data shape without rows
    dtype: np.dtype or Python data type
        Data type
    """

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    def __repr__(self):
        return "Array <shape={}, dtype={}>".format(self.shape, self.dtype)


class ClassLabel(Array):
    """
    Label for classification tasks. It specifies class names

    Parameters
    ----------
    num_classes: int, optional
        Number of classes.
    names: list of str, optional
        Class names.
    names_file: str, optional
        File path where every line represents a single class name.
    multilabel: bool, optional
        Whether or not the label is multilabel
    dtype: numpy.dtype or Python data type, optional
        Data type. By default, int
    """

    def __init__(self, num_classes=None, names=None, names_file=None, multilabel=False, dtype=None):
        if sum(bool(a) for a in (num_classes, names, names_file)) != 1:
            raise ValueError("Only a single argument of ClassLabel(num_classes, names, names_file) should be provided.")

        if names is not None:
            self.names = names
            self.num_classes = len(self.names)
        elif names_file is not None:
            self.names = read_lines(names_file)
            self.num_classes = len(self.names)
        elif num_classes is not None:
            self.num_classes = num_classes
            self.names = None
        else:
            raise ValueError("A single argument of ClassLabel() should be provided")

        if multilabel:
            shape = [self.num_classes]
        else:
            shape = []

        self.multilabel = multilabel

        if dtype is None:
            dtype = int

        if self.names is not None:
            self._name_to_idx = dict(zip(self.names, range(self.num_classes)))

        super().__init__(shape=shape, dtype=dtype)

    def str2int(self, str_value):
        """
        Convert class name to index

        Parameters
        ----------
        str_value: str
            Class name

        Returns
        -------
        index: int
            Class index
        """
        if self.names is None:
            return None

        return self._name_to_idx[str_value]

    def int2str(self, int_value):
        """
        Convert index to class name

        Parameters
        ----------
        int_value: int
            Class index

        Returns
        --------
            Class name
        """
        if self.names is None:
            return None

        return self.names[int_value]

    def __repr__(self):
        return "ClassLabel <shape={}, dtype={}, classes={}>".format(self.shape, self.dtype, self.num_classes)
