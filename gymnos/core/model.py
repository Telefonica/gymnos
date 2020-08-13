#
#
#   Model
#
#

import logging

from copy import deepcopy

from .. import models

from ..utils.py_utils import cached_property, drop

logger = logging.getLogger(__name__)


class Model:
    """
    Parameters
    ----------
    model: dict
        Model type and their parameters with the structure ``{"type", **parameters}``
    training: dict, optional
        Dictionnary with training parameters (``fit`` or ``fit_generator`` arguments)

    Examples
    --------

    .. code-block:: py

        Model(
            model={
                "type": "dogs_vs_cats",
                "input_shape": [100, 100, 1],
                "classes": 4
            },
            training={
                "epochs": 10,
                "batch_size": 32
            }
        )
    """  # noqa: E501

    def __init__(self, model, training=None):
        model = model or {}
        training = training or {}

        self.training = training

        self.model_spec = deepcopy(model)

    @property
    def parameters(self):
        return drop(self.model_spec, "type")

    @cached_property
    def model(self):
        return models.load(**self.model_spec)

    def to_dict(self):
        return dict(
            model=self.model_spec,
            training=self.training
        )
