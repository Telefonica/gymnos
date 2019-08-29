import gymnos
import pytest
import numbers
import numpy as np
import pandas as pd

from gymnos.datasets.mte import MTE
from gymnos.datasets.boston_housing import BostonHousing
from gymnos.datasets.dogs_vs_cats import DogsVsCats
from gymnos.datasets.imdb import IMDB
from gymnos.datasets.rock_paper_scissors import RockPaperScissors
from gymnos.datasets.synthetic_digits import SyntheticDigits
from gymnos.datasets.tiny_imagenet import TinyImagenet
from gymnos.datasets.data_usage_test import DataUsageTest
from gymnos.datasets.unusual_data_usage_test import UnusualDataUsageTest

from gymnos.datasets.dataset import ClassLabel
from gymnos.services.download_manager import DownloadManager


ARRAY_LIKE_TYPES = [numbers.Number, frozenset, list, set, tuple, np.ndarray, pd.Series, pd.DataFrame]


def test_load():
    dataset = gymnos.datasets.load("boston_housing")

    assert isinstance(dataset, gymnos.datasets.dataset.Dataset)

    with pytest.raises(ValueError):
        _ = gymnos.datasets.load("dummy")


@pytest.mark.slow
@pytest.mark.parametrize("dataset", [
    MTE(),
    RockPaperScissors(),
    BostonHousing(),
    DogsVsCats(),
    IMDB(),
    SyntheticDigits(),
    TinyImagenet(),
    DataUsageTest(),
    UnusualDataUsageTest()
])
def test_samples(dataset, tmp_path):
    dl_manager = DownloadManager(download_dir=str(tmp_path / "Downloads"))
    dataset.download_and_prepare(dl_manager)

    sample = dataset[0]
    assert len(sample) == 2

    assert isinstance(sample[0], tuple(ARRAY_LIKE_TYPES + [str]))
    assert isinstance(sample[1], tuple(ARRAY_LIKE_TYPES + [str]))

    np_array_samples = np.array([dataset[0][0], dataset[1][0]])

    assert np_array_samples.shape[0] == 2

    info = dataset.info()

    if hasattr(sample[0], "shape"):
        assert tuple(info.features.shape) == tuple(sample[0].shape)
    if hasattr(sample[1], "shape"):
        assert tuple(info.labels.shape) == tuple(sample[1].shape)

    if hasattr(sample[0], "dtype"):
        assert np.issubdtype(sample[0].dtype, info.features.dtype)

    if isinstance(info.labels, ClassLabel):
        assert np.issubdtype(info.labels.dtype, np.integer)
        assert isinstance(sample[1], (int, np.integer))

    assert len(dataset) > 0
