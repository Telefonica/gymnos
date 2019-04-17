#
#
#   Dataset
#
#

import os

from pydoc import locate

from ..utils.io_utils import read_from_json
from ..preprocessors import Pipeline
from ..logger import get_logger

DATASETS_IDS_TO_MODULES_PATH = os.path.join(os.path.dirname(__file__), "..", "var", "datasets.json")
PREPROCESSORS_IDS_TO_MODULES_PATH = os.path.join(os.path.dirname(__file__), "..", "var", "preprocessors.json")


class DatasetSamples:

    def __init__(self, train=None, test=0.25):
        self.test = test
        if train is None:
            self.train = 1 - test
        else:
            self.train = train

        if (self.test + self.train < 1.0):
            get_logger(prefix=self).warning("Using only {}% of total data".format(self.train + self.test))


class Dataset:

    def __init__(self, name, samples=None, preprocessors=None, seed=None, shuffle=True, cache_dir=None):
        samples = samples or {}
        preprocessors = preprocessors or []

        self.name = name
        self.seed = seed
        self.shuffle = shuffle

        self.samples = DatasetSamples(**samples)

        DatasetClass = self.__retrieve_dataset_from_id(name)
        self.dataset = DatasetClass(cache_dir=cache_dir)

        self.preprocessor_pipeline = Pipeline()
        for preprocessor_config in preprocessors:
            PreprocessorClass = self.__retrieve_preprocessor_from_type(preprocessor_config.pop("type"))
            preprocessor = PreprocessorClass(**preprocessor_config)
            self.preprocessor_pipeline.add(preprocessor)


    def __retrieve_dataset_from_id(self, dataset_id):
        datasets_ids_to_modules = read_from_json(DATASETS_IDS_TO_MODULES_PATH)
        dataset_loc = datasets_ids_to_modules[dataset_id]
        return locate(dataset_loc)


    def __retrieve_preprocessor_from_type(self, preprocessor_type):
        preprocessors_ids_to_modules = read_from_json(PREPROCESSORS_IDS_TO_MODULES_PATH)
        preprocessor_loc = preprocessors_ids_to_modules[preprocessor_type]
        return locate(preprocessor_loc)
