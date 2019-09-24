#
#
#   Test model
#
#

import pytest
import gymnos

from gymnos.core.model import Model


def test_model_instance():
    model = Model(dict(
        type="dogs_vs_cats_cnn",
        input_shape=[120, 120, 3]
    ))

    assert isinstance(model.model, gymnos.models.dogs_vs_cats_cnn.DogsVsCatsCNN)

    with pytest.raises(ValueError):
        model = Model(dict(type="dummy"))
