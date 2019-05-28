#
#
#   Data
#
#

import math
import keras
import numpy as np
import pandas as pd

from tqdm import tqdm


class Subset:
    """
    Subset of a dataset at specified indices.

    Parameters
    ----------
        dataset: sequence
            Dataset implementing ``__geitem__`` and ``__len__`` methods.
        indices: list of ints
            Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)


def default_collate_func(samples):
    """
    Merge list of samples to form a mini-batch.

    Parameters
    ----------
        samples: list or objects

    Returns
    -------
    np.array or pd.DataFrame
        If type for ``samples`` is ``pd.Series``, return ``pd.DataFrame``, else, return
        ``np.array``.
    """
    sample = samples[0]
    if isinstance(sample, (pd.Series, pd.DataFrame)):
        return pd.concat(samples, axis=1)
    else:
        return np.array(samples)


class DataLoader(keras.utils.Sequence):
    # we inherit from keras Sequence to make it work with Keras but keras.Sequence doesn't provide
    # any additional functionality, it only defines abstract methods

    """
    Data loader. Combines a dataset and provides iterators over the dataset.

    Parameters
    ----------
    dataset: sequence
        Dataset implementing ``__geitem__`` and ``__len__`` methods.
    batch_size: int, optional
        How many samples per batch to load
    drop_last: bool, optional
        Set to ``True`` to drop the last incomplete batch, if the dataset size is not divisible by the ``batch_size``.
    collate_func: func, optional
        Function to merge a list of samples to form a mini-batch.
    transform_func: func, optional
        Function to apply to every batch before returning them
    verbose: bool, optional
        Show progress bar while fetching items
    """

    def __init__(self, dataset, batch_size=1, drop_last=False, collate_func=None, transform_func=None, verbose=False):
        self.dataset = dataset
        self.drop_last = drop_last
        self.batch_size = batch_size
        self.transform_func = transform_func
        self.verbose = verbose

        if collate_func is None:
            collate_func = default_collate_func

        self.collate_func = collate_func

    def __getitem__(self, index):
        batch_index_start = index * self.batch_size
        batch_index_end   = batch_index_start + self.batch_size
        if batch_index_end > len(self.dataset):
            if self.drop_last:
                raise IndexError()

            batch_index_end = len(self.dataset)

        features = []
        labels = []

        iterator = range(batch_index_start, batch_index_end)

        if self.verbose:
            iterator = tqdm(iterator)

        for index in iterator:
            x, y = self.dataset[index]
            features.append(x)
            labels.append(y)

        features_batch = self.collate_func(features)
        labels_batch = self.collate_func(labels)
        if self.transform_func:
            features_batch, labels_batch = self.transform_func([features_batch, labels_batch])

        return features_batch, labels_batch

    def __len__(self):
        if self.drop_last:
            round_op = math.floor
        else:
            round_op = math.ceil

        return round_op(len(self.dataset) / self.batch_size)
