#
#
#   Dataset
#
#

import h5py
import pickle
import numpy as np

from tqdm import tqdm
from abc import ABCMeta, abstractmethod

from ..utils.data import DataLoader
from ..utils.io_utils import read_lines


class Dataset(metaclass=ABCMeta):
    """
    Base class for all Gymnos datasets.

    You need to implement the following methods: ``download_and_prepare``, ``info``, ``__getitem__`` and ``__len__``.
    """
    @abstractmethod
    def info(self):
        """
        Returns info about dataset features and labels

        Returns
        -------
        dataset_info: DatasetInfo
        """

    @abstractmethod
    def download_and_prepare(self, dl_manager):
        """
        Download files and prepare instance for future calls to ``__getitem__`` and ``__len__``.

        Parameters
        ----------
        dl_manager: DownloadManager
            Download Manager to download files.
        """

    @abstractmethod
    def __getitem__(self, index):
        """
        Returns single row of data

        Parameters
        ----------
        index: int
            Row index to retrieve


        Returns
        -------
        row: np.array or pd.Series
            Single row of data
        """

    @abstractmethod
    def __len__(self):
        """
        Returns number of rows

        Returns
        -------
        int
            Number of samples
        """

    def as_numpy(self):
        features = []
        labels = []
        for index in range(len(self)):
            X, y = self[index]
            features.append(X)
            labels.append(y)

        features = np.array(features)
        labels = np.array(labels)

        return features, labels

    def to_hdf5(self, file_path, features_key="features", labels_key="labels", info_key="info", chunk_size=None,
                compression="gzip", compression_opts=None, force=False):
        """
        Export dataset to HDF5 file.

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
        if chunk_size is None:
            chunk_size = len(self)

        data_loader = DataLoader(self, batch_size=chunk_size)

        info = self.info()

        mode = "w" if force else "x"

        with h5py.File(file_path, mode) as h5f:
            labels_shape = [len(self)] + info.labels.shape
            features_shape = [len(self)] + info.features.shape

            labels_dtype = info.labels.dtype
            features_dtype = info.features.dtype

            if features_dtype == str:
                features_dtype = h5py.special_dtype(vlen=str)

            features = h5f.create_dataset(features_key, shape=features_shape, dtype=features_dtype,
                                          compression=compression, compression_opts=compression_opts)
            labels = h5f.create_dataset(labels_key, shape=labels_shape, compression=compression,
                                        dtype=labels_dtype, compression_opts=compression_opts)

            features.attrs[info_key] = np.string_(pickle.dumps(info.features))
            labels.attrs[info_key] = np.string_(pickle.dumps(info.labels))

            for index, (X, y) in enumerate(tqdm(data_loader)):
                start = index * data_loader.batch_size
                end = start + data_loader.batch_size
                features[start:end] = X
                labels[start:end] = y


class HDF5Dataset(Dataset):
    """
    Create dataset from HDF5 file.

    Parameters
    ----------
    file_path: str
        HDF5 dataset file path.
    features_key: str
        Key to load features.
    labels_key: str
        Key to load labels
    info_key: str
        Key to load info
    """

    def __init__(self, file_path, features_key="features", labels_key="labels", info_key="info"):
        self.features_key = features_key
        self.labels_key = labels_key
        self.info_key = info_key

        self.data = h5py.File(file_path, mode="r")

    def info(self):
        features_info = pickle.loads(self.data[self.features_key].attrs[self.info_key])
        labels_info = pickle.loads(self.data[self.labels_key].attrs[self.info_key])
        return DatasetInfo(
            features=features_info,
            labels=labels_info
        )

    def download_and_prepare(self, dl_manager):
        """
        It does nothing.
        """
        pass

    def __getitem__(self, index):
        return (
            self.data[self.features_key][index],
            self.data[self.labels_key][index]
        )

    def __len__(self):
        return len(self.data[self.features_key])  # len(features) == len(labels)


class DatasetInfo:
    """
    Dataset info.

    Parameters
    ----------
    features: Array
        Info about features shape and dtype
    labels: Array
        Info about labels shape and dtype
    """

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __repr__(self):
        return "Features <{}>, Labels <{}>".format(self.features, self.labels)


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
        return self.names.index(str_value)

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
        return self.names[int_value]

    def __repr__(self):
        return "ClassLabel <shape={}, dtype={}, classes={}>".format(self.shape, self.dtype, self.num_classes)
