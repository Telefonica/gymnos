#
#
#   Data test
#
#

import pytest
import numpy as np
from gymnos.utils.data import Subset, DataLoader
from gymnos.datasets import Dataset


class TestSubset:

    def test_getitem(self):
        sequence = np.arange(3)
        indices = list(reversed(sequence))
        subset = Subset(sequence, indices)

        assert subset[0] == 2
        assert subset[1] == 1
        assert subset[2] == 0

        assert len(subset) == len(sequence)


    def test_getitem_2(self):
        sequence = np.arange(5)
        indices = sequence
        subset = Subset(sequence, indices)

        for i in indices:
            assert subset[i] == i

        assert len(subset) == len(sequence)



class NumericDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def info(self):
        pass

    def download_and_prepare(self, dl_manager):
        pass

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class TestDataLoader:

    @pytest.fixture
    def dataset(self):
        x = np.random.randint(0, 10, (6, 3))
        y = np.random.randint(0, 2, 6)

        return NumericDataset(x=x, y=y)

    def test_getitem(self, dataset):
        loader = DataLoader(dataset, batch_size=3, drop_last=False, transform_func=None, verbose=False)

        assert len(loader) == 2

        assert loader[0][0].shape == dataset.x[:3].shape
        assert loader[0][1].shape == dataset.y[:3].shape

        assert np.array_equal(loader[0][0], dataset.x[:3])
        assert np.array_equal(loader[0][1], dataset.y[:3])

        assert np.array_equal(loader[1][0], dataset.x[3:])
        assert np.array_equal(loader[1][1], dataset.y[3:])

    def test_getitem_2(self, dataset):
        loader = DataLoader(dataset, batch_size=len(dataset), drop_last=False)

        assert len(loader) == 1

        assert np.array_equal(loader[0][0], dataset.x)
        assert np.array_equal(loader[0][1], dataset.y)

    def test_getitem_3(self, dataset):
        def squared(data):
            return data[0]**2, data[1]

        loader = DataLoader(dataset, batch_size=2, transform_func=squared)

        assert len(loader) == 3
        assert len(loader[0][0]) == 2
        assert len(loader[0][1]) == 2

        assert np.array_equal(loader[0][0], dataset.x[:2]**2)
        assert np.array_equal(loader[0][1], dataset.y[:2])


    def test_getitem_4(self, dataset):
        loader = DataLoader(dataset, batch_size=4, drop_last=True)

        assert len(loader) == 1

        assert np.array_equal(loader[0][0], dataset.x[:4])
        assert np.array_equal(loader[0][1], dataset.y[:4])
