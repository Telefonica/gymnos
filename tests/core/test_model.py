#
#
#   Test model
#
#

import pytest
import gymnos

from gymnos.core import Model


def test_model_instance():
    model = Model("dogs_vs_cats_cnn", parameters=dict(input_shape=[120, 120, 3]))

    assert isinstance(model.model, gymnos.models.dogs_vs_cats_cnn.DogsVsCatsCNN)

    with pytest.raises(ValueError):
        model = Model("dummy")
