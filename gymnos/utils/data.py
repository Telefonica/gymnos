#
#
#   Data
#
#

import time
import math
import logging
import warnings
import pandas as pd
import numpy as np

from tqdm import tqdm
from collections.abc import Iterable


logger = logging.getLogger(__name__)


def is_sequence(obj):
    return hasattr(obj, "__getitem__") and hasattr(obj, "__len__")


def gen_intervals(*splits):
    """
    Generate intervals from splits.

    Examples
    ---------
    >>> gen_intervals(0.8, 0.2)
    [[0.0, 0.8], [0.8, 1.0]]
    >>> gen_intervals(0.3, 0.2, 0.5)
    [[0.0, 0.3], [0.3, 0.5], [0.5, 1.0]]

    Returns
    --------
    intervals: list
        List with intervals
    """
    intervals = []
    current_end_interval = 0.0
    for split in splits:
        start = current_end_interval
        current_end_interval = start + split
        intervals.append([start, current_end_interval])
    return intervals


def gen_random_seed():
    """
    Generate random seed

    Returns
    ---------
    int
    """
    t = int(time.time() * 1000.0)
    return ((t & 0xff000000) >> 24) + ((t & 0x00ff0000) >> 8) + ((t & 0x0000ff00) << 8) + ((t & 0x000000ff) << 24)


def collate(samples):
    """
    Merge list of samples to form a mini-batch.

    Parameters
    ----------
        samples: list of objects

    Returns
    -------
    np.ndarray or pd.DataFrame
    """
    assert len(samples) > 0

    if isinstance(samples[0], pd.Series):
        return pd.concat(samples, axis=1).T

    if isinstance(samples[0], pd.DataFrame):
        return pd.concat(samples)

    if isinstance(samples[0], str):
        return pd.Series(samples)  # prefer pandas over numpy for array of strings

    return np.array(samples)


def safe_indexing(X, indices):
    """
    Return items or rows from X using indices.
    Allows simple indexing of lists or arrays.
    Parameters
    ----------
    X : array-like, sparse-matrix, list, pandas.DataFrame, pandas.Series.
        Data from which to sample rows or items.
    indices : array-like of int
        Indices according to which X will be subsampled.
    Returns
    -------
    subset
        Subset of X on first axis
    Notes
    -----
    CSR, CSC, and LIL sparse matrices are supported. COO sparse matrices are
    not supported.
    """
    if hasattr(X, "iloc"):
        # Work-around for indexing with read-only indices in pandas
        # indices = indices if indices.flags.writeable else indices.copy()
        # Pandas Dataframes and Series
        try:
            return X.iloc[indices]
        except ValueError:
            # Cython typed memoryviews internally used in pandas do not support
            # readonly buffers.
            warnings.warn("Copying input dataframe for slicing.")
            return X.copy().iloc[indices]
    elif hasattr(X, "shape"):
        if hasattr(X, 'take') and (hasattr(indices, 'dtype') and
                                   indices.dtype.kind == 'i'):
            # This is often substantially faster than X[indices]
            return X.take(indices, axis=0)
        else:
            return X[indices]
    else:
        return [X[idx] for idx in indices]


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


class DataLoader:
    """
    Data loader. Combines a dataset and provides iterators over the dataset.

    Parameters
    ----------
    dataset: Sequence
        Sequence implementing ``getitem__`` and ``__len__`` methods
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
        if collate_func is None:
            collate_func = collate

        self.dataset = dataset
        self.drop_last = drop_last
        self.batch_size = batch_size
        self.transform_func = transform_func
        self.verbose = verbose
        self.collate_func = collate_func

    def __getitem__(self, index):
        """
        Returns batch of samples.

        Returns
        ----------
        batch: np.ndarray
        """
        batch_index_start = index * self.batch_size
        batch_index_end = batch_index_start + self.batch_size
        if batch_index_end > len(self.dataset):
            if self.drop_last:
                raise IndexError()

            batch_index_end = len(self.dataset)

        iterator = range(batch_index_start, batch_index_end)

        if self.verbose:
            iterator = tqdm(iterator)

        batch_samples = [self.dataset[index] for index in iterator]
        batch_samples = [self.collate_func(samples) for samples in zip(*batch_samples)]

        if self.transform_func is not None:
            batch_samples = self.transform_func(batch_samples)

        return batch_samples

    def __iter__(self):
        """
        Create a generator that iterate over the DataLoader.
        """
        for item in (self[i] for i in range(len(self))):
            yield item

    def __len__(self):
        """
        Returns number of batches.
        """
        round_op = math.floor if self.drop_last else math.ceil
        return round_op(len(self.dataset) / self.batch_size)


class IterableDataLoader:
    """
    Data loader. Combines a dataset and provides iterators over the dataset.

    Parameters
    ----------
    dataset: Iterable
        Iterable implementing ``__iter__`` and ``__len__`` methods
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
            collate_func = collate

        self.collate_func = collate_func

    def __iter__(self):
        """
        Iterate over DataLoader.

        Yields
        ------
        batch: np.ndarray
            Batch of samples.
        """
        def prepare_batch(batch):
            batch = [self.collate_func(samples) for samples in zip(*batch)]
            if self.transform_func is not None:
                batch = self.transform_func(batch)
            return batch

        iterator = self.dataset

        if self.verbose:
            iterator = tqdm(iterator)

        batch_samples = []

        for row in iterator:
            batch_samples.append(row)

            if len(batch_samples) >= self.batch_size:
                yield prepare_batch(batch_samples)

                batch_samples = []

        if not self.drop_last and (len(batch_samples) > 0):
            yield prepare_batch(batch_samples)

    def __len__(self):
        """
        Returns number of batches
        """
        round_op = math.floor if self.drop_last else math.ceil
        return round_op(len(self.dataset) / self.batch_size)


class SplitIterator(Iterable):
    """
    Iterator for split datasets.

    Parameters
    ------------
    iterable: Iterable
        Iterable implementing ``__iter__`` and ``__len__``.
    start: float
        Fraction to start splitting
    end: float
        Fraction to end splitting
    shuffle: bool
        Whether or not shuffle iterator
    random_state: int
        Random seed
    """

    def __init__(self, dataset, start, end, shuffle=False, random_state=0):
        self.dataset = dataset
        self.start = start
        self.end = end
        self.shuffle = shuffle
        self.random_state = random_state

        self.cache_count = None

    def _count_random_iterator(self):
        """
        """
        count = 0

        random_generator = np.random.RandomState(self.random_state)
        for _ in range(len(self.dataset)):
            if self.start <= random_generator.rand() < self.end:
                count += 1

        return count

    def _random_iterator(self):
        """
        """
        count = 0

        random_generator = np.random.RandomState(self.random_state)
        for row in self.dataset:
            if self.start <= random_generator.rand() < self.end:
                yield row

                count += 1

        self.cache_count = count

    def _sequential_iterator(self):
        """
        """
        iterable_length = len(self.dataset)
        start_idx = int(iterable_length * self.start)
        end_idx = int(iterable_length * self.end)

        for idx, row in enumerate(self.dataset):
            if start_idx <= idx < end_idx:
                yield row

    def __iter__(self):
        """
        """
        if self.shuffle:
            yield from self._random_iterator()
        else:
            yield from self._sequential_iterator()

    def __len__(self):
        """
        """
        if not self.shuffle:
            return math.floor(len(self.iterable) * (self.end - self.start))

        if self.cache_count is not None:
            return self.cache_count

        self.cache_count = self._count_random_iterator()

        return self.cache_count


def split_iterator(iterable, splits, shuffle=True, random_state=None):
    """
    Split an iterator into multiple splits

    Parameters
    ----------
    iterable: Iterable
        Iterable implementing ``__getitem__`` and ``__len__``.
    splits: list or tuple of floats or ints
        Fraction or number of samples for each iterable.
    shuffle: bool
        Whether or not shuffle dataset
    random_state: int, optional
        Random seed. If not provided, seed will be generated automatically.

    Notes
    --------
    Due to technical reasons, if ``shuffle`` parameter is a truthy value then the splits will be a probability not an
    exact ratio.

    The algorithm works as follows:

        If shuffle is ``True``:

            We iterate over rows. For each row we generate a random number and if the random numbes lies within
            an interval created

    Returns
    ---------
    list of SplitIterator
    """
    logger.warning("Splits for iterable datasets will be just a probability so we can't guarantee "
                   "the precision for each split")

    if random_state is None:
        random_state = gen_random_seed()

    def to_proba(split):
        """
        Convert to probability if split is number of samples.
        """
        if 0 < split < 1:
            return split
        return split / len(iterable)

    splits = [to_proba(split) for split in splits]

    split_ranges = gen_intervals(*splits)

    def build_iterator(split_range):
        return SplitIterator(iterable, split_range[0], split_range[1], shuffle=shuffle,
                             random_state=random_state)

    return list(map(build_iterator, split_ranges))


def split_sequence(sequence, splits, shuffle=True, random_state=None):
    """
    Split a sequence into multiple splits

    Parameters
    -----------
    sequence: Sequence
        Sequence implementing ``__getitem__`` and ``__len__``.
    splits: list or tuple of floats or ints
        List of splits
    shuffle: bool
        Whether or not shuffle dataset
    random_state: int, optional
        Random seed. If not provided, seed will be generated automatically.

    Returns
    --------
    list of Subset
    """
    indices = np.arange(len(sequence))
    if shuffle:
        indices = np.random.RandomState(seed=random_state).permutation(indices)

    start = 0
    subsets = []
    for split in splits:
        num_samples = split
        if 0 < split < 1.0:
            num_samples = math.floor(len(sequence) * split)

        end = start + num_samples
        subset = Subset(sequence, indices[start:end])
        subsets.append(subset)
        start = end

    return subsets


def split_spark_dataframe(dataframe, splits, shuffle=True, random_state=None):
    """
    Split Spark DataFrame

    Parameters
    ------------
    dataframe: pyspark.sql.DataFrame
        Spark DataFrame
    splits: list of floats
        List of probabilities (between 0 and 1)
    shuffle: bool
        Whether or not shuffle dataset. Not shuffling is currently not implemented
    random_state: int, optional
        Seed for splitting
    """
    if shuffle:
        return dataframe.randomSplit(splits, seed=random_state)  # FIXME: handle non probabilities
    else:
        raise NotImplementedError()


def forever_generator(iterator):
    """
    Create an infinite generator from an interator
    """
    while True:
        yield from iterator
