import gymnos
import pytest


def test_load():
    model = gymnos.models.load("dogs_vs_cats_cnn", input_shape=[10, 10, 3])

    assert isinstance(model, gymnos.models.model.Model)

    with pytest.raises(ValueError):
        _ = gymnos.models.load("dummy")
