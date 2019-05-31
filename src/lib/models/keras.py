#
#
#   Keras
#
#

import os

from pydoc import locate
from keras import models

from .model import Model
from .mixins import KerasMixin
from ..utils.io_utils import read_from_json


KERAS_LAYERS_IDS_TO_MODULES_PATH = os.path.join(os.path.dirname(__file__), "..", "var", "keras", "layers.json")
KERAS_METRICS_IDS_TO_MODULES_PATH = os.path.join(os.path.dirname(__file__), "..", "var", "keras", "metrics.json")
KERAS_OPTIMIZERS_IDS_TO_MODULES_PATH = os.path.join(os.path.dirname(__file__), "..", "var", "keras", "optimizers.json")
KERAS_APPLICATIONS_IDS_TO_MODULES_PATH = os.path.join(os.path.dirname(__file__), "..", "var", "keras",
                                                      "applications.json")


class Keras(KerasMixin, Model):
    """
    Model to build Keras sequentials from a dictionnary that defines the network architecture.

    Parameters
    ----------
    sequential: list of dict
        TODO
    compilation: dict
        TODO

    Note
    ----
    This model requires one-hot encoded labels.
    """

    def __init__(self, sequential, compilation):
        loss, optimizer, metrics = self.__build_compilation_from_config(**compilation)
        self.model = self.__build_sequential_from_config(sequential)
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)


    def __build_compilation_from_config(self, loss, optimizer, metrics=None):
        metrics = metrics or []

        if isinstance(optimizer, dict):
            optimizers_ids_to_modules = read_from_json(KERAS_OPTIMIZERS_IDS_TO_MODULES_PATH)
            OptimizerClass = locate(optimizers_ids_to_modules[optimizer.pop("type")])
            optimizer = OptimizerClass(**optimizer)

        metrics_funcs = []
        metrics_ids_to_modules = read_from_json(KERAS_METRICS_IDS_TO_MODULES_PATH)
        for metric_name in metrics:
            if metric_name in metrics_ids_to_modules:
                metric_func = locate(metrics_ids_to_modules[metric_name])
                metrics_funcs.append(metric_func)
            else:
                metrics_funcs.append(metric_name)

        return loss, optimizer, metrics_funcs


    def __build_sequential_from_config(self, sequential_config):
        sequential = models.Sequential()
        for layer_config in sequential_config:
            layer_type = layer_config.pop("type")
            if layer_type == "application":
                application_ids_to_modules = read_from_json(KERAS_APPLICATIONS_IDS_TO_MODULES_PATH)
                LayerClass = locate(application_ids_to_modules[layer_config.pop("application")])
                layer = LayerClass(**layer_config)
            else:
                layers_ids_to_modules = read_from_json(KERAS_LAYERS_IDS_TO_MODULES_PATH)
                LayerClass = locate(layers_ids_to_modules[layer_type])
                layer = LayerClass(**layer_config)

            sequential.add(layer)

        return sequential
