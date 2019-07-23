#
#
#   Test dataset
#
#

import pytest
import gymnos.preprocessors
import gymnos.data_augmentors

from gymnos.core.dataset import Dataset
from gymnos.datasets import BostonHousing

DATASET_NAME = "boston_housing"


def test_dataset_instance():
    dataset = Dataset(DATASET_NAME)
    assert isinstance(dataset.dataset, BostonHousing)

    with pytest.raises(ValueError):
        Dataset("dummy")


def test_dataset_samples():
    samples = dict(train=0.3, test=0.7)
    dataset = Dataset(DATASET_NAME, samples=samples)

    assert dataset.samples.train == 0.3
    assert dataset.samples.test == 0.7

    samples = dict(train=5, test=10)
    dataset = Dataset(DATASET_NAME, samples=samples)

    assert dataset.samples.train == 5
    assert dataset.samples.test == 10

    with pytest.raises(ValueError):
        samples = dict(train=1.0, test=0.9)
        dataset = Dataset(DATASET_NAME, samples=samples)


def test_dataset_preprocessors():
    preprocessor_spec_1 = dict(
        type="divide",
        factor=255.0
    )
    preprocessor_spec_2 = dict(
        type="replace",
        from_val=1.0,
        to_val=0.0
    )

    dataset = Dataset(DATASET_NAME, preprocessors=[preprocessor_spec_1, preprocessor_spec_2])

    assert isinstance(dataset.preprocessors, gymnos.preprocessors.Pipeline)

    assert len(dataset.preprocessors) == 2

    for preprocessor in dataset.preprocessors.preprocessors:
        assert isinstance(preprocessor, gymnos.preprocessors.Preprocessor)

    dataset = Dataset(DATASET_NAME)

    assert isinstance(dataset.preprocessors, gymnos.preprocessors.Pipeline)

    assert len(dataset.preprocessors) == 0

    preprocessor_spec = dict(
        type="dummy",
        arg="hello"
    )

    with pytest.raises(ValueError):
        dataset = Dataset(DATASET_NAME, preprocessors=[preprocessor_spec])


def test_dataset_data_augmentors():
    data_augmentor_spec_1 = dict(
        type="invert",
        probability=0.4
    )

    data_augmentor_spec_2 = dict(
        type="distort",
        probability=0.4,
        grid_width=120,
        grid_height=100,
        magnitude=10
    )

    dataset = Dataset(DATASET_NAME, data_augmentors=[data_augmentor_spec_1, data_augmentor_spec_2])

    assert isinstance(dataset.data_augmentors, gymnos.data_augmentors.Pipeline)

    assert len(dataset.data_augmentors) == 2

    dataset = Dataset(DATASET_NAME)

    assert isinstance(dataset.data_augmentors, gymnos.data_augmentors.Pipeline)

    assert len(dataset.data_augmentors) == 0

    dummy = dict(
        type="dummy",
        arg="hello"
    )
    with pytest.raises(ValueError):
        dataset = Dataset(DATASET_NAME, data_augmentors=[dummy])