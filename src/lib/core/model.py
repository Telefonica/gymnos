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

    def __init__(self, name, parameters=None, description=None):
        parameters = parameters or {}

        ModelClass = self.__retrieve_model_from_name(name)
        self.model = ModelClass(**parameters)
        self.parameters = parameters
        self.description = description


    def __retrieve_model_from_name(self, name):
        models_ids_to_modules = read_from_json(MODELS_IDS_TO_MODULES_PATH)
        return locate(models_ids_to_modules[name])
