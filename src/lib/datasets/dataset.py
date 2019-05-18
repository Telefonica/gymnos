#
#
#   Dataset
#
#

import numpy as np
from ..logger import get_logger


def requires_download(func):
    def wrapped(self, *args, **kwargs):
        if not self._downloaded:
            raise RuntimeError("You must download your model before using it.")
        return func(self, *args, **kwargs)

    return wrapped


class Dataset:
    """
    Base class for all Gymnos datasets.

    You need to implement the following methods: ``download_and_prepare``, ``info`` and ``_load``.
    """

    def __init__(self):
        self.logger = get_logger(prefix=self)
        self._downloaded = False

    def _info(self):
        raise NotImplementedError()

    def info(self):
        return self._info()

    def _download_and_prepare(self, dl_manager):
        raise NotImplementedError()

    def download_and_prepare(self, dl_manager):
        self._downloaded = True
        return self._download_and_prepare(dl_manager)

    def _load(self):
        raise NotImplementedError()

    @requires_download
    def load(self):
        return self._load()

    def _select(self, start=0, stop=None):
        raise NotImplementedError()

    @requires_download
    def select(self, start=0, stop=None):
        return self.select(start=start, stop=stop)

    def _nsamples(self):
        raise NotImplementedError()

    @requires_download
    def nsamples(self):
        return self._nsamples()


class DatasetInfo:

    def __init__(self, features, labels):
        if not isinstance(features, Tensor):
            features = Tensor(shape=[], dtype=features)

        if not isinstance(labels, (Tensor)):
            labels = Tensor(shape=[], dtype=labels)

        self.features = features
        self.labels = labels


class Tensor:

    def __init__(self, shape, dtype=None):
        self.shape = shape
        self.dtype = dtype

    def __str__(self):
        return "Tensor <shape={}, dtype={}>".format(self.shape, self.dtype)


class ClassLabel(Tensor):

    def __init__(self, num_classes=None, names=None, names_file=None):
        super().__init__(shape=[], dtype=np.int32)

        if sum(bool(a) for a in (num_classes, names, names_file)) != 1:
            raise ValueError("Only a single argument of ClassLabel() should be provided.")

        if names is not None:
            self.names = names
            self.num_classes = len(self.names)
        elif names_file is not None:
            self.names = self.__load_names_from_file(names_file)
            self.num_classes = len(self.names)
        elif num_classes is not None:
            self.num_classes = num_classes
            self.names = None
        else:
            raise ValueError("A single argument of ClassLabel() should be provided")

    def __load_names_from_file(self, names_file):
        with open(names_file) as archive:
            names = [line.rstrip('\n') for line in archive]

        return names

    def str2int(self, str_value):
        return self.names.index(str_value)

    def int2str(self, int_value):
        return self.names[int_value]

    def __str__(self):
        return "Tensor <shape={}, dtype={} classes={}>".format(self.shape, self.dtype, self.num_classes)
