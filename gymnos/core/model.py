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
    Examples
    --------

    .. code-block:: py

        Model(
            name="data_usage_holt_winters",
            parameters={
                "beta": 0.029,
                "alpha": 0.5
            }
        )
    """  # noqa: E501

    def __init__(self, name, parameters=None, training=None):
        training = training or {}
        parameters = parameters or {}

        self.name = name
        self.parameters = parameters

        self.training = training

        self.model = models.load(name, **deepcopy(parameters))

    def to_dict(self):
        return dict(
            name=self.name,
            parameters=self.parameters,
            training=self.training
        )
