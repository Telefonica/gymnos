#
#
#   Model
#
#

import os

from pydoc import locate

from keras import models, layers

from ..models import KerasModel
from ..utils.io_utils import read_from_json

LAYERS_IDS_TO_MODULES_PATH = os.path.join(os.path.dirname(__file__), "..", "var", "layers.json")
MODELS_IDS_TO_MODULES_PATH = os.path.join(os.path.dirname(__file__), "..", "var", "models.json")
OPTIMIZERS_IDS_TO_MODULES_PATH = os.path.join(os.path.dirname(__file__), "..", "var", "optimizers.json")
APPLICATIONS_IDS_TO_MODULES_PATH = os.path.join(os.path.dirname(__file__), "..", "var", "applications.json")


class ModelCompilation:

    def __init__(self, loss, optimizer, metrics=None):
        self.loss = loss
        self.metrics = metrics

        if isinstance(optimizer, str):
            OptimizerClass = self.__retrieve_optimizer_from_id(optimizer)
            self.optimizer = OptimizerClass()
        else:
            OptimizerClass = self.__retrieve_optimizer_from_id(optimizer.pop("type"))
            self.optimizer = OptimizerClass(**optimizer)

    def __retrieve_optimizer_from_id(self, optimizer_id):
        optimizers_ids_to_modules = read_from_json(OPTIMIZERS_IDS_TO_MODULES_PATH)
        optimizer_loc = optimizers_ids_to_modules[optimizer_id]
        return locate(optimizer_loc)


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

        self.description = description

        self.compilation = None

        if isinstance(self.model, KerasModel):
            self.compilation = ModelCompilation(**compilation)
            self.model.compile(loss=self.compilation.loss, optimizer=self.compilation.optimizer,
                               metrics=self.compilation.metrics)


    def __retrieve_model_from_id(self, model_id):
        models_ids_to_modules = read_from_json(MODELS_IDS_TO_MODULES_PATH)
        model_loc = models_ids_to_modules[model_id]
        return locate(model_loc)


    def __retrieve_layer_from_type(self, layer_type):
        layers_ids_to_modules = read_from_json(LAYERS_IDS_TO_MODULES_PATH)
        layer_loc = layers_ids_to_modules[layer_type]
        return locate(layer_loc)

    def __retrieve_application_from_application_id(self, application_id):
        application_ids_to_modules = read_from_json(APPLICATIONS_IDS_TO_MODULES_PATH)
        application_loc = application_ids_to_modules[application_id]
        return locate(application_loc)

    def __build_keras_model_from_network(self, input_shape, network):
        input_layer = layers.Input(shape=input_shape)

        output_layer = input_layer
        for layer_config in network:
            layer_type = layer_config.pop("type")
            if layer_type == "application":
                LayerClass = self.__retrieve_application_from_application_id(layer_config.pop("application"))
            else:
                LayerClass = self.__retrieve_layer_from_type(layer_type)

            keras_layer = LayerClass(**layer_config)

            output_layer = keras_layer(output_layer)

        keras_model = models.Model(inputs=input_layer, outputs=output_layer)

        return KerasModel(input_shape, keras_model)
