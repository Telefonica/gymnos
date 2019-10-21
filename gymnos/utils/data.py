#
#
#   Data
#
#

import time
import math
import numpy as np

from tqdm import tqdm
from collections.abc import Iterable


def gen_intervals(*splits):
    """
    Generate intervals from splits.

    Examples
    ---------
    >>> gen_intervals(0.8, 0.2)
    [[0.0, 0.8], 0.8, 1.0]
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


def default_collate_func(samples):
    """
    Merge list of samples to form a mini-batch.

    Parameters
    ----------
        samples: list of objects

    Returns
    -------
    np.ndarray
    """
    return np.array(samples)


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
        self.dataset = dataset
        self.drop_last = drop_last
        self.batch_size = batch_size
        self.transform_func = transform_func
        self.verbose = verbose

        if collate_func is None:
            collate_func = default_collate_func

        self.collate_func = collate_func

    def __getitem__(self, index):
        """
        Returns batch of samples.

        Returns
        ----------
        batch: np.ndarray
        """
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
            collate_func = default_collate_func

        self.collate_func = collate_func

    def __iter__(self):
        """
        Iterate over DataLoader.

        Yields
        ------
        batch: np.ndarray
            Batch of samples.
        """
        def prepare_batch(features, labels):
            features_batch = self.collate_func(features)
            labels_batch = self.collate_func(labels)

            if self.transform_func is not None:
                features_batch, labels_batch = self.transform_func([features_batch, labels_batch])

            return features_batch, labels_batch

        features_batch, labels_batch = [], []

        iterator = self.dataset

        if self.verbose:
            iterator = tqdm(iterator, total=len(iterator))

        for x, y in iterator:
            features_batch.append(x)
            labels_batch.append(y)

            if len(features_batch) == self.batch_size:
                features_collated_batch, labels_collated_batch = prepare_batch(features_batch, labels_batch)

                yield features_collated_batch, labels_collated_batch

                features_batch, labels_batch = [], []

        if not self.drop_last and (len(features_batch) > 0):
            features_collated_batch, labels_collated_batch = prepare_batch(features_batch, labels_batch)
            yield features_collated_batch, labels_collated_batch

    def __len__(self):
        """
        Returns number of batches
        """
        round_op = math.floor if self.drop_last else math.ceil
        return round_op(len(self.dataset) / self.batch_size)


class SplitIterator(Iterable):
    """
    Parameters
    ------------
    iterable: Iterable
        Iterable implementing ``__iter__`` and ``__len__``.
    start: float

    end: float

    shuffle: bool
        Whether or not shuffle iterator
    random_state: int
        Random seed
    """

    def __init__(self, iterable, start, end, shuffle=False, random_state=0):
        self.iterable = iterable
        self.start = start
        self.end = end
        self.shuffle = shuffle
        self.random_state = random_state

        self.cache_count = None

    def count_random_iterator(self):
        """
        """
        count = 0

        random_generator = np.random.RandomState(self.random_state)
        for _ in range(len(self.iterable)):
            if self.start <= random_generator.rand() < self.end:
                count += 1

        return count

    def random_iterator(self):
        """
        """
        count = 0

        random_generator = np.random.RandomState(self.random_state)
        for row in self.iterable:
            if self.start <= random_generator.rand() < self.end:
                yield row

                count += 1

        self.cache_count = count

    def sequential_iterator(self):
        """
        """
        iterable_length = len(self.iterable)
        start_idx = int(iterable_length * self.start)
        end_idx = int(iterable_length * self.end)

        for idx, row in enumerate(self.iterable):
            if start_idx <= idx < end_idx:
                yield row

    def __iter__(self):
        """
        """
        if self.shuffle:
            yield from self.random_iterator()
        else:
            yield from self.sequential_iterator()

    def __len__(self):
        """
        """
        if not self.shuffle:
            return math.floor(len(self.iterable) * (self.end - self.start))

        if self.cache_count is not None:
            return self.cache_count

        self.cache_count = self.count_random_iterator()

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


def forever_generator(iterator):
    """
    Create an infinite generator from an interator
    """
    while True:
        yield from iterator
