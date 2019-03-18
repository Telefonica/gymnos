#
#
#   Model
#
#

import os

from pydoc import locate

from keras import models, layers

from .models import KerasModel
from .utils.io_utils import read_from_json

LAYERS_IDS_TO_MODULES_PATH = os.path.join(os.path.dirname(__file__), "var", "layers.json")
MODELS_IDS_TO_MODULES_PATH = os.path.join(os.path.dirname(__file__), "var", "models.json")
ESTIMATORS_IDS_TO_MODULES_PATH = os.path.join(os.path.dirname(__file__), "var", "estimators.json")


class ModelCompilation:

    def __init__(self, loss, optimizer, metrics=None):
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics


class Model:

    def __init__(self, input_shape, description=None, compilation=None, model=None,
                 network=None, hyperparameters=None):

        hyperparameters = hyperparameters or {}

        if model is not None:
            ModelClass = self.__retrieve_model_from_id(model)
            self.model = ModelClass(input_shape, **hyperparameters)
        elif network is not None:
            self.model = self.__build_keras_model_from_network(input_shape, network)
        else:
            raise ValueError("You need to provide either a model, or a network")

        if compilation is not None:
            self.compilation = ModelCompilation(**compilation)
        else:
            self.compilation = None

        self.description = description


    def __retrieve_model_from_id(self, model_id):
        models_ids_to_modules = read_from_json(MODELS_IDS_TO_MODULES_PATH)
        model_loc = models_ids_to_modules[model_id]
        return locate(model_loc)


    def __retrieve_layer_from_type(self, layer_type):
        layers_ids_to_modules = read_from_json(LAYERS_IDS_TO_MODULES_PATH)
        layer_loc = layers_ids_to_modules[layer_type]
        return locate(layer_loc)


    def __retrieve_estimator_from_id(self, estimator_id):
        estimators_ids_to_modules = read_from_json(ESTIMATORS_IDS_TO_MODULES_PATH)
        estimator_loc = estimators_ids_to_modules[estimator_id]
        return locate(estimator_loc)

    def __build_keras_model_from_network(self, input_shape, network):
        input_layer = layers.Input(shape=input_shape)

        output_layer = input_layer
        for layer_config in network:
            LayerClass = self.__retrieve_layer_from_type(layer_config.pop("type"))
            keras_layer = LayerClass(**layer_config)

            output_layer = keras_layer(output_layer)

        keras_model = models.Model(inputs=input_layer, outputs=output_layer)

        return KerasModel(input_shape, keras_model)
