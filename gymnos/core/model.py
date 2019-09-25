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

    def __init__(self, model, parameters=None, training=None):
        model = model or {}
        training = training or {}
        parameters = parameters or {}

        self.parameters = parameters

        self.training = training

        self.model_spec = deepcopy(model)

        self.model = models.load(**model)

    def to_dict(self):
        return dict(
            model=self.model_spec,
            training=self.training
        )
