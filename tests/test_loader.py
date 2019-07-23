#
#
#   Test init
#
#

import os
import json
import pytest
import gymnos
import gymnos.models

from pydoc import locate


def test_load_dataset_error():
    with pytest.raises(ValueError):
        _ = gymnos.load(dataset="dummy")


def test_load_dataset():
    dataset = gymnos.load(dataset="boston_housing")

    assert isinstance(dataset, gymnos.datasets.Dataset)

    with pytest.raises(ValueError):
        _ = gymnos.load(dataset="dummy")


def test_load_model():
    model = gymnos.load(model="mte_nn", input_shape=[120])

    assert isinstance(model, gymnos.models.Model)

    with pytest.raises(ValueError):
        _ = gymnos.load(model="dummy")


def test_load_preprocessor():
    preprocessor = gymnos.load(preprocessor="divide", factor=255.0)

    assert isinstance(preprocessor, gymnos.preprocessors.Preprocessor)

    assert preprocessor.factor == 255.0

    with pytest.raises(ValueError):
        _ = gymnos.load(preprocessor="dummy")


def test_load_tracker():
    tracker = gymnos.load(tracker="tensorboard")

    assert isinstance(tracker, gymnos.trackers.Tracker)

    with pytest.raises(ValueError):
        _ = gymnos.load(tracker="dummy")


def test_load_data_augmentor():
    data_augmentor = gymnos.load(data_augmentor="invert", probability=1)

    assert isinstance(data_augmentor, gymnos.data_augmentors.DataAugmentor)

    with pytest.raises(ValueError):
        _ = gymnos.load(data_augmentor="dummy")


@pytest.mark.parametrize(("filename", "subclass"), [
    ("datasets.json", gymnos.datasets.Dataset),
    ("models.json", gymnos.models.Model),
    ("trackers.json", gymnos.trackers.Tracker),
    ("preprocessors.json", gymnos.preprocessors.Preprocessor),
    ("data_augmentors.json", gymnos.data_augmentors.DataAugmentor)
])
def test_var_modules(filename, subclass):
    var_path = os.path.join("gymnos", "var", filename)
    with open(var_path) as fp:
        var = json.load(fp)

    for name, module_path in var.items():
        module = locate(module_path)
        assert issubclass(module, subclass)
