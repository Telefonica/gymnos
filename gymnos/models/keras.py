#
#
#   Keras
#
#

import os

from pydoc import locate
from keras import models, layers

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

    def __init__(self, input_shape, sequential, optimizer, loss=None, metrics=None):

        optimizer = self.__build_optimizer_from_config(optimizer)
        metrics = self.__build_metrics_from_config(metrics)
        self.model = self.__build_sequential_from_config(input_shape, sequential)
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def __build_optimizer_from_config(self, optimizer):
        if isinstance(optimizer, dict):
            optimizers_ids_to_modules = read_from_json(KERAS_OPTIMIZERS_IDS_TO_MODULES_PATH)
            OptimizerClass = locate(optimizers_ids_to_modules[optimizer.pop("type")])
            optimizer = OptimizerClass(**optimizer)

        return optimizer


    def __build_metrics_from_config(self, metrics):
        metrics_funcs = []
        metrics_ids_to_modules = read_from_json(KERAS_METRICS_IDS_TO_MODULES_PATH)
        for metric_name in metrics:
            if metric_name in metrics_ids_to_modules:
                metric_func = locate(metrics_ids_to_modules[metric_name])
                metrics_funcs.append(metric_func)
            else:
                metrics_funcs.append(metric_name)

        return metrics_funcs


    def __build_sequential_from_config(self, input_shape, sequential_config):
        input_layer = layers.Input(input_shape)

        output_layer = input_layer
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

            output_layer = layer(output_layer)

        model = models.Model(inputs=[input_layer], outputs=[output_layer])

        return model
