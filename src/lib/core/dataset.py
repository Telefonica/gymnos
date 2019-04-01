#
#
#   Dataset
#
#

import os

from pydoc import locate

from ..transformers import TransformerStack
from ..utils.io_utils import read_from_json
from ..preprocessors import PreprocessorStack

TRANSFORMERS_IDS_TO_MODULES_PATH = os.path.join(os.path.dirname(__file__), "..", "var", "transformers.json")
DATASETS_IDS_TO_MODULES_PATH = os.path.join(os.path.dirname(__file__), "..", "var", "datasets.json")
PREPROCESSORS_IDS_TO_MODULES_PATH = os.path.join(os.path.dirname(__file__), "..", "var", "preprocessors.json")


class Dataset:

    def __init__(self, name, preprocessors=None, transformers=None, cache_dir=None):
        preprocessors = preprocessors or []
        transformers = transformers or []

        self.name = name

        DatasetClass = self.__retrieve_dataset_from_id(name)
        self.dataset = DatasetClass(cache_dir=cache_dir)

        self.preprocessor_stack = PreprocessorStack()
        for preprocessor_config in preprocessors:
            PreprocessorClass = self.__retrieve_preprocessor_from_type(preprocessor_config.pop("type"))
            preprocessor = PreprocessorClass(**preprocessor_config)
            self.preprocessor_stack.add(preprocessor)

        self.transformer_stack = TransformerStack()
        for transformer_config in transformers:
            TransformerClass = self.__retrieve_transformer_from_type(transformer_config.pop("type"))
            transformer = TransformerClass(**transformer_config)
            self.transformer_stack.add(transformer)


    def __retrieve_dataset_from_id(self, dataset_id):
        datasets_ids_to_modules = read_from_json(DATASETS_IDS_TO_MODULES_PATH)
        dataset_loc = datasets_ids_to_modules[dataset_id]
        return locate(dataset_loc)


    def __retrieve_preprocessor_from_type(self, preprocessor_type):
        preprocessors_ids_to_modules = read_from_json(PREPROCESSORS_IDS_TO_MODULES_PATH)
        preprocessor_loc = preprocessors_ids_to_modules[preprocessor_type]
        return locate(preprocessor_loc)


    def __retrieve_transformer_from_type(self, transformer_type):
        transformers_ids_to_modules = read_from_json(TRANSFORMERS_IDS_TO_MODULES_PATH)
        transformer_loc = transformers_ids_to_modules[transformer_type]
        return locate(transformer_loc)
