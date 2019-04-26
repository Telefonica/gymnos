#
#
#   Model
#
#

import os

from pydoc import locate

from ..utils.io_utils import read_from_json

MODELS_IDS_TO_MODULES_PATH = os.path.join(os.path.dirname(__file__), "..", "var", "models.json")


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
    description: str, optional
        Optional description of the model
    Examples
    --------

    .. code-block:: py

        Model(
            name="data_usage_holt_winters",
            description="Holt Winters with low parameters",
            parameters={
                "beta": 0.029,
                "alpha": 0.5
            }
        )
    """

    def __init__(self, name, parameters=None, description=None):
        parameters = parameters or {}

        self.name = name
        self.parameters = parameters
        self.description = description

        ModelClass = self.__retrieve_model_from_name(name)
        self.model = ModelClass(**parameters)


    def __retrieve_model_from_name(self, name):
        models_ids_to_modules = read_from_json(MODELS_IDS_TO_MODULES_PATH)
        return locate(models_ids_to_modules[name])
