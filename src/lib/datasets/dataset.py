#
#
#   Dataset
#
#

import h5py
import pickle


class Dataset:
    """
    Base class for all Gymnos datasets.

    You need to implement the following methods: ``download_and_prepare``, ``info``, ``__getitem__`` and ``__len__`.
    """

    def info(self):
        raise NotImplementedError()

    def download_and_prepare(self, dl_manager):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()


class HDF5Dataset(Dataset):

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
        pass

    def __getitem__(self, index):
        return (
            self.data[self.features_key][index],
            self.data[self.labels_key][index]
        )

    def __len__(self):
        return len(self.data[self.features_key])  # len(features) == len(labels)


class DatasetInfo:

    def __init__(self, features, labels):
        if not isinstance(features, Tensor):
            features = Tensor(shape=[], dtype=features)

        if not isinstance(labels, (Tensor)):
            labels = Tensor(shape=[], dtype=labels)

        self.features = features
        self.labels = labels


class Tensor:

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    def __str__(self):
        return "Tensor <shape={}, dtype={}>".format(self.shape, self.dtype)


def _read_lines(file_path):
    with open(file_path) as archive:
        names = [line.rstrip('\n') for line in archive]

    return names


class ClassLabel(Tensor):

    def __init__(self, num_classes=None, names=None, names_file=None, multilabel=False, dtype=None):
        if sum(bool(a) for a in (num_classes, names, names_file)) != 1:
            raise ValueError("Only a single argument of ClassLabel() should be provided.")

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
        return self.names.index(str_value)

    def int2str(self, int_value):
        return self.names[int_value]

    def __str__(self):
        return "ClassLabel <shape={}, dtype={}, classes={}>".format(self.shape, self.dtype, self.num_classes)
