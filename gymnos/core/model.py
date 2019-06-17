#
#
#   Model
#
#

import os
import logging

from ..utils.io_utils import import_from_json

MODELS_IDS_TO_MODULES_PATH = os.path.join(os.path.dirname(__file__), "..", "var", "models.json")

logger = logging.getLogger(__name__)


class Model:
    """
    Parameters
    ----------

    name: str
        Name of the model.
        The current available models are the following:

        - ``"dogs_vs_cats_cnn"``: :class:`lib.models.dogs_vs_cats_cnn.DogsVsCatsCNN`,
        - ``"data_usage_linear_regression"``: :class:`lib.models.data_usage_linear_regression.DataUsageLinearRegression`,
        - ``"data_usage_holt_winters"``: :class:`lib.models.data_usage_holt_winters.DataUsageHoltWinters`,
        - ``"unusual_data_usage_weighted_thresholds"``: :class:`lib.models.unusual_data_usage_weighted_thresholds.UnusualDataUsageWT`,
        - ``"fashion_mnist_nn"``: :class:`lib.models.fashion_mnist_nn.FashionMnistNN`,
        - ``"keras"``: :class:`lib.models.keras.Keras`,
        - ``"mte_nn"``: :class:`lib.models.mte_nn.MTENN`
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

    def __init__(self, name, parameters=None):
        parameters = parameters or {}

        self.name = name
        self.parameters = parameters

        logger.debug("Importing model {}".format(name))
        ModelClass = import_from_json(MODELS_IDS_TO_MODULES_PATH, name)
        self.model = ModelClass(**parameters)
