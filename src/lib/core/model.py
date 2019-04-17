#
#
#   Model
#
#

import os

from pydoc import locate

from keras import models

from ..logger import get_logger
from ..models import KerasModel
from ..utils.io_utils import read_from_json

KERAS_LAYERS_IDS_TO_MODULES_PATH = os.path.join(os.path.dirname(__file__), "..", "var", "keras", "layers.json")
MODELS_IDS_TO_MODULES_PATH = os.path.join(os.path.dirname(__file__), "..", "var", "models.json")
KERAS_METRICS_IDS_TO_MODULES_PATH = os.path.join(os.path.dirname(__file__), "..", "var", "keras", "metrics.json")
KERAS_OPTIMIZERS_IDS_TO_MODULES_PATH = os.path.join(os.path.dirname(__file__), "..", "var", "keras", "optimizers.json")
KERAS_APPLICATIONS_IDS_TO_MODULES_PATH = os.path.join(os.path.dirname(__file__), "..", "var", "keras",
                                                      "applications.json")


class ModelCompilation:

    def __init__(self, loss, optimizer, metrics=None):
        metrics = metrics or []

        self.loss = loss

        if isinstance(optimizer, str):
            OptimizerClass = self.__retrieve_optimizer_from_id(optimizer)
            self.optimizer = OptimizerClass()
        else:
            OptimizerClass = self.__retrieve_optimizer_from_id(optimizer.pop("type"))
            self.optimizer = OptimizerClass(**optimizer)

        self.metrics = [self.__retrieve_metric_from_id(metric) for metric in metrics]


    def __retrieve_optimizer_from_id(self, optimizer_id):
        optimizers_ids_to_modules = read_from_json(KERAS_OPTIMIZERS_IDS_TO_MODULES_PATH)
        optimizer_loc = optimizers_ids_to_modules[optimizer_id]
        return locate(optimizer_loc)

    def __retrieve_metric_from_id(self, metric_id):
        metrics_ids_to_modules = read_from_json(KERAS_METRICS_IDS_TO_MODULES_PATH)
        metric_loc = metrics_ids_to_modules.get(metric_id)
        if metric_loc is None:
            return metric_id

        return locate(metric_loc)


class Model:

    def __init__(self, description=None, compilation=None, name=None,
                 network=None, parameters=None):

        self.logger = get_logger(prefix=self)

        self.parameters = parameters or {}

        if name is not None:
            ModelClass = self.__retrieve_model_from_id(name)
            self.model = ModelClass(**self.parameters)
        elif network is not None:
            self.model = self.__build_keras_model_from_network(network)
        else:
            raise ValueError("You need to provide either a model, or a network")

        self.description = description

        self.compilation = None

        if isinstance(self.model, KerasModel):
            self.logger.info("Compiling Keras model")
            self.compilation = ModelCompilation(**compilation)
            self.model.compile(loss=self.compilation.loss, optimizer=self.compilation.optimizer,
                               metrics=self.compilation.metrics)


    def __retrieve_model_from_id(self, model_id):
        models_ids_to_modules = read_from_json(MODELS_IDS_TO_MODULES_PATH)
        model_loc = models_ids_to_modules[model_id]
        return locate(model_loc)


    def __retrieve_layer_from_type(self, layer_type):
        layers_ids_to_modules = read_from_json(KERAS_LAYERS_IDS_TO_MODULES_PATH)
        layer_loc = layers_ids_to_modules[layer_type]
        return locate(layer_loc)

    def __retrieve_application_from_application_id(self, application_id):
        application_ids_to_modules = read_from_json(KERAS_APPLICATIONS_IDS_TO_MODULES_PATH)
        application_loc = application_ids_to_modules[application_id]
        return locate(application_loc)

    def __build_keras_model_from_network(self, network):
        self.logger.info("Building Keras model from network specification")

        sequential = models.Sequential()
        for layer_config in network:
            layer_type = layer_config.pop("type")
            if layer_type == "application":
                LayerClass = self.__retrieve_application_from_application_id(layer_config.pop("application"))
                keras_layer = LayerClass(**layer_config)
            else:
                LayerClass = self.__retrieve_layer_from_type(layer_type)
                keras_layer = LayerClass(**layer_config)

            sequential.add(keras_layer)

        return KerasModel(sequential)
