#
#
#   Dataset
#
#

import h5py
import pickle

from abc import ABCMeta, abstractmethod


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

    def __str__(self):
        return "Array <shape={}, dtype={}>".format(self.shape, self.dtype)


def _read_lines(file_path):
    """
    Read file lines

    Parameters
    -----------
    file_path: str
        File path to read lines.

    Returns
    -------
    num_lines: int
        Number of lines
    """
    with open(file_path) as archive:
        names = [line.rstrip('\n') for line in archive]

    return names


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
            self.names = _read_lines(names_file)
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

    def __str__(self):
        return "ClassLabel <shape={}, dtype={}, classes={}>".format(self.shape, self.dtype, self.num_classes)
