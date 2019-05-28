#
#
#   Dataset
#
#

from sys import getsizeof


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

    @property
    def memory_usage(self):
        sample = self[0]
        features_nbytes = getsizeof(sample[0])
        labels_nbytes = getsizeof(sample[1])
        return (features_nbytes * len(self)) + (labels_nbytes * len(self))


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

    def __init__(self, num_classes=None, names=None, names_file=None, multilabel=False):
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

        if multilabel:
            shape = [self.num_classes]
        else:
            shape = []

        self.multilabel = multilabel

        super().__init__(shape=shape, dtype=int)

    def __load_names_from_file(self, names_file):
        with open(names_file) as archive:
            names = [line.rstrip('\n') for line in archive]

        return names

    def str2int(self, str_value):
        return self.names.index(str_value)

    def int2str(self, int_value):
        return self.names[int_value]

    def __str__(self):
        return "Tensor <shape={}, dtype={}, classes={}>".format(self.shape, self.dtype, self.num_classes)
