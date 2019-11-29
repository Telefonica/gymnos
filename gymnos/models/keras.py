#
#
#   Keras
#
#

from tensorflow.keras import models, layers

from .model import Model
from .utils.keras_modules import import_keras_module
from .mixins import KerasClassifierMixin, KerasRegressorMixin

from ..utils.py_utils import drop


class BaseKeras(Model):
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

    def __init__(self, input_shape, sequential, optimizer, loss, metrics=None):
        metrics = metrics or []

        optimizer = self.__build_optimizer_from_config(optimizer)
        metrics = self.__build_metrics_from_config(metrics)
        self.model = self.__build_sequential_from_config(input_shape, sequential)
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def __build_optimizer_from_config(self, optimizer):
        if isinstance(optimizer, dict):
            cls = import_keras_module(optimizer["type"], "optimizers")
            optimizer = cls(**drop(optimizer, "type"))

        return optimizer

    def __build_metrics_from_config(self, metrics):
        metrics_funcs = []

        for metric_name in metrics:
            try:
                metric_func = import_keras_module(metric_name, "metrics")
                metrics_funcs.append(metric_func)
            except ValueError:
                metrics_funcs.append(metric_name)

        return metrics_funcs

    def __build_sequential_from_config(self, input_shape, sequential_config):
        input_layer = layers.Input(input_shape)

        output_layer = input_layer
        for layer_config in sequential_config:
            if layer_config["type"] == "application":
                cls = import_keras_module(layer_config["application"], "applications")
                layer = cls(**drop(layer_config, "application"))
            else:
                cls = import_keras_module(layer_config["type"], "layers")
                layer = cls(**drop(layer_config, "type"))

            output_layer = layer(output_layer)

        model = models.Model(inputs=[input_layer], outputs=[output_layer])

        return model


class KerasClassifier(KerasClassifierMixin, BaseKeras):
    """
    Keras classifier
    """


class KerasRegressor(KerasRegressorMixin, BaseKeras):
    """
    Keras regressor
    """
