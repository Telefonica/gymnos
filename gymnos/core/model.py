#
#
#   Model
#
#

import logging

from copy import deepcopy

from .. import models

logger = logging.getLogger(__name__)


class Model:
    """
    Parameters
    ----------

    name: str
        Model name.
    parameters: dict, optional
        Parameters associated with the model
    model: dict, optional
        Model type and their parameters with the structure ``{"type", **parameters}``

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

        self.model = models.load(**model)

    @property
    def parameters(self):
        return {key: val for key, val in self.model_spec.items() if key != "type"}

    def to_dict(self):
        return dict(
            model=self.model_spec,
            training=self.training
        )
