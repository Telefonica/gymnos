#
#
#   Create new model/dataset/preprocessor/tracker
#
#

import os
import re
import sys
import inspect
import argparse

from pprint import pprint
from collections import OrderedDict

from lib.utils.io_utils import read_from_json, save_to_json
from lib.core.dataset import Dataset
from lib.core.model import Model
from lib.core.experiment import Experiment
from lib.core.tracking import Tracking
from lib.core.training import Training


SNAKE_CASE_TEST_RE = re.compile(r'^[a-z]+([a-z\d]+_|_[a-z\d]+|[a-z\d]+)+[a-z\d]+$')
SNAKE_CASE_TEST_DASH_RE = re.compile(r'^[a-z]+([a-z\d]+-|-[a-z\d]+)+[a-z\d]+$')
SNAKE_CASE_REPLACE_RE = re.compile(r'(_)([a-z\d])')
SNAKE_CASE_REPLACE_DASH_RE = re.compile('(-)([a-z\d])')

VAR_FILES_DIR = os.path.join("lib", "var")
DATASETS_VAR_FILE_PATH = os.path.join(VAR_FILES_DIR, "datasets.json")
MODELS_VAR_FILE_PATH = os.path.join(VAR_FILES_DIR, "models.json")
TRACKERS_VAR_FILE_PATH = os.path.join(VAR_FILES_DIR, "trackers.json")
PREPROCESSORS_VAR_FILE_PATH = os.path.join(VAR_FILES_DIR, "preprocessors.json")


def is_string(obj):
    """
    Checks if an object is a string.
    :param obj: Object to test.
    :return: True if string, false otherwise.
    :rtype: bool
    """
    try:  # basestring is available in python 2 but missing in python 3!
        return isinstance(obj, basestring)
    except NameError:
        return isinstance(obj, str)


def is_full_string(string):
    """
    Check if a string is not empty (it must contains at least one non space character).
    :param string: String to check.
    :type string: str
    :return: True if not empty, false otherwise.
    """
    return is_string(string) and string.strip() != ""


def is_snake_case(string, separator="_"):
    """
    Checks if a string is formatted as snake case.
    A string is considered snake case when:
    * it"s composed only by lowercase letters ([a-z]), underscores (or provided separator) \
    and optionally numbers ([0-9])
    * it does not start/end with an underscore (or provided separator)
    * it does not start with a number
    :param string: String to test.
    :type string: str
    :param separator: String to use as separator.
    :type separator: str
    :return: True for a snake case string, false otherwise.
    :rtype: bool
    """
    if is_full_string(string):
        re_map = {
            "_": SNAKE_CASE_TEST_RE,
            "-": SNAKE_CASE_TEST_DASH_RE
        }
        re_template = "^[a-z]+([a-z\d]+{sign}|{sign}[a-z\d]+|[a-z\d]+)+[a-z\d]+$"
        r = re_map.get(separator, re.compile(re_template.format(sign=re.escape(separator))))
        return bool(r.search(string))
    return False


def snake_case_to_camel(string, upper_case_first=True, separator='_'):
    """
    Convert a snake case string into a camel case one.
    (The original string is returned if is not a valid snake case string)
    :param string: String to convert.
    :type string: str
    :param upper_case_first: True to turn the first letter into uppercase (default).
    :type upper_case_first: bool
    :param separator: Sign to use as separator (default to "_").
    :type separator: str
    :return: Converted string
    :rtype: str
    """
    if not is_string(string):
        raise TypeError('Expected string')
    if not is_snake_case(string, separator):
        return string
    re_map = {
        '_': SNAKE_CASE_REPLACE_RE,
        '-': SNAKE_CASE_REPLACE_DASH_RE
    }
    r = re_map.get(separator, re.compile('({sign})([a-z\d])'.format(sign=re.escape(separator))))
    string = r.sub(lambda m: m.group(2).upper(), string)
    if upper_case_first:
        return string[0].upper() + string[1:]
    return string


DATASET_TYPE = "dataset"
MODEL_TYPE = "model"
PREPROCESSOR_TYPE = "preprocessor"
TRACKER_TYPE = "tracker"
EXPERIMENT_TYPE = "experiment"

DATASET_FILE_STR = """
#
#
#   {dataset_name}
#
#

from .dataset import Dataset, DatasetInfo, ClassLabel


class {dataset_name}:
    \"""
    {TODO}: Description of my dataset.
    \"""

    def info(self):
        # {TODO}: Specifies the DatasetInfo object
        return DatasetInfo(
            features=...,
            labels=...
        )

    def download_and_prepare(self, dl_manager):
        # {TODO}: download any file you will need later in the __getitem__ and __len__ function

    def __getitem__(self, given):
        # {TODO}: Get dataset item/s. Given can be a slice object or an int. Called after download_and_prepare.

    def __len__(self):
        # {TODO}: Dataset length. Called after download_and_prepare
"""

MODEL_FILE_STR = """
#
#
#   {model_name}
#
#

from .model import Model


class {model_name}(Model):
    \"""
    {TODO}: Description of my model.
    \"""

    def __init__(self, **parameters):
        # {TODO}: Define and initialize model parameters.


    def fit(self, X, y, **training_parameters):
        # {TODO}: Fit model to training data.

    def fit_generator(self, X, y, **training_parameters):
        # {OPTIONAL}: Fit model to training generator. Write method if your model supports incremental learning
        raise NotImplementedError()
    def predict(self, X):
        # {TODO}: Predict classes/values using features.

    def predict_proba(self, X):
        # {OPTIONAL}: Predict probabilities using features. Write method if your model is a probabilistic model
        raise NotImplementedError()

    def save(self, save_dir):
        # {TODO}: Save model to save_dir.

    def restore(self, save_dir):
        # {TODO}: Restore model from save_dir.
"""

PREPROCESSOR_FILE_STR = """
#
#
#   {preprocessor_name}
#
#

from .preprocessor import Preprocessor


class {preprocessor_name}(Preprocessor):
    \"""
    #{TODO}: Description of my preprocessor.
    \"""

    def __init__(self, **parameters):
        # {TODO}: Define and initialize model parameters

    def fit(self, X, y=None):
        # {TODO}: Fit preprocessor to training data.

    def fit_generator(self, generator):
        # {OPTIONAL}: Fit preprocessor to training generator. Only if your preprocessor supports incremental learning
        raise NotImplementedError()

    def transform(self, X):
        # {TODO}: Preprocess data
"""

TRACKER_FILE_STR = """
#
#
#   {tracker_name}
#
#

from .tracker import Tracker


class {tracker_name}(Tracker):
    \"""
    #{TODO}: Description of my tracker
    \"""

    def __init__(self, **parameters):
        # {TODO}: Define and initialize tracker parameters

    def start(run_name, logdir):
        # {OPTIONAL}: Initialize tracker
        pass

    def add_tag(self, tag):
        # {OPTIONAL}: Add tag
        pass

    def log_asset(self, name, file_path):
        # {OPTIONAL}: Log asset
        pass

    def log_image(self, name, file_path):
        # {OPTIONAL}: Log image
        pass

    def log_figure(self, name, figure):
        # {OPTIONAL}: Log Matplotlib figure
        pass

    def log_metric(self, name, value, step=None):
        # {OPTIONAL}: Log metric
        pass

    def log_param(self, name, value, step=None):
        # {OPTIONAL}: Log parameter
        pass

    def log_other(self, name, value):
        # {OPTIONAL}: Log other
        pass

    def end(self):
        # {OPTIONAL}: Called when the experiment is finished
        pass
"""

EXPERIMENT_FILE_STR = """
{{
    "experiment": {{
        "name": null,  // {OPTIONAL} (str): Experiment name. This name will be used as execution directory where we store model, pipeline, logs, etc ... If not specified, default experiment name is defined in config/preferences.json
        "description": null,  // {OPTIONAL} (str): Experiment description
        "tags": [  // {OPTIONAL} (list of str): Experiment tags

        ]
    }},
    "dataset": {{
        "name": xxxxxx,  // {TODO} (str): dataset identifier.  To see available datasets, check lib/var/datasets.json 
        "samples": {{   // {OPTIONAL} (dict<str: int or float>): Split dataset into train and test subset. If float, it specifies a ratio, e.g 0.75 -> 75%, if int it specifies number of samples. Train and test samples must be > 0.
            "train": 0.75,
            "test": 0.25
        }},
        "preprocessors": [    // {OPTIONAL} (list of dicts): Define preprocessors in the following format: {{"type": <preprocessor_name>, **preprocessor_parameters}}, e.g {{"type": "divide", "factor": 255.0}}. To see available preprocessors, check lib/var/preprocessors.json 

        ],
        "seed": null,  // {OPTIONAL} (int): Seed for train/test split. If null, random seed is chosen
        "shuffle": true,  // {OPTIONAL} (bool): Whether or not shuffle dataset
        "one_hot": false, // {OPTIONAL} (bool): Whether or not one-hot encode labels. Some models require one-hot encoded labels, check documentation.
        "chunk_size": null  // {OPTIONAL} (int): Defines the chunk size in which the dataset will be read.. It may reduce memory usage. By default, it loads dataset into memory
    }},
    "model": {{
        "name": xxxxxx,  // {TODO} (str): Model identifier. To see available models, check lib/var/models.json 
        "parameters": {{  // {OPTIONAL} (str): Parameters for model constructor

        }}
    }},
    "training": {{  // {OPTIONAL} (str): Parameters for model "fit" method or "fit_generator" method if dataset.chunk_size is not null

    }},
    "tracking": {{
        "log_model_params": true,  // {OPTIONAL} (bool): Whether or not log model parameters
        "log_model_metrics": true,  // {OPTIONAL} (bool): Whether or not log metrics
        "log_training_params": true,  // {OPTIONAL} (bool): Whether or not log training parameters
        "params": {{  // {OPTIONAL} (dict): Additional parameters to log

        }},
        "trackers": [  // {OPTIONAL} (list of dicts): Define trackers {{"type": <tracker_name>, **tracker_parameters}}, e.g {{"type": "mlflow", "source_name": "gymnos_gpu"}}  To see available trackers, check lib/var/trackers.json 

        ]
    }}
}}
""" # noqa: E501

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("type", type=str, choices=[DATASET_TYPE, MODEL_TYPE, PREPROCESSOR_TYPE, TRACKER_TYPE,
                                                   EXPERIMENT_TYPE])
    parser.add_argument("--name", type=str, required=True, help="Name to generate files (snake case)")
    args = parser.parse_args()

    if not is_snake_case(args.name):
        error_str = ("Name must be snake case. "
                     "A string is considered snake case when: \n" +
                     "    * it's composed only by lowercase letters ([a-z]), underscores " +
                     " and optionally numbers ([0-9])\n" +
                     "    * it does not start/end with an underscore\n" +
                     "    * it does not start with a number\n")
        raise ValueError(error_str)


    camel_name = snake_case_to_camel(args.name)

    if args.type == EXPERIMENT_TYPE:
        file_str = EXPERIMENT_FILE_STR.format(TODO="TODO", OPTIONAL="OPTIONAL")
        file_path = os.path.join("experiments", args.name + ".json")

        with open(file_path, "x") as archive:
            archive.write(file_str)

        print("The following files have been created: {}".format(file_path))

        sys.exit()


    if args.type == DATASET_TYPE:
        dirname = "datasets"
        file_str = DATASET_FILE_STR.format(dataset_name=camel_name, TODO="TODO({})".format(camel_name),
                                           OPTIONAL="OPTIONAL({})".format(camel_name))
    elif args.type == MODEL_TYPE:
        dirname = "models"
        file_str = MODEL_FILE_STR.format(model_name=camel_name, TODO="TODO({})".format(camel_name),
                                         OPTIONAL="OPTIONAL({})".format(camel_name))
    elif args.type == PREPROCESSOR_TYPE:
        dirname = "preprocessors"
        file_str = PREPROCESSOR_FILE_STR.format(preprocessor_name=camel_name, TODO="TODO({})".format(camel_name),
                                                OPTIONAL="OPTIONAL({})".format(camel_name))
    elif args.type == TRACKER_TYPE:
        dirname = "trackers"
        file_str = PREPROCESSOR_FILE_STR.format(preprocessor_name=camel_name, TODO="TODO({})".format(camel_name),
                                                OPTIONAL="OPTIONAL({})".format(camel_name))

    file_dir = os.path.join("lib", dirname)
    file_path = os.path.join(file_dir, args.name + ".py")
    init_file_path = os.path.join(file_dir, "__init__.py")
    var_file_path = os.path.join("lib", "var", dirname + ".json")

    with open(file_path, "x") as archive:
        archive.write(file_str)

    with open(init_file_path, "a") as archive:
        archive.write("from .{} import {}\n".format(args.name, camel_name))

    var_data = read_from_json(var_file_path)
    var_data[args.name] = ".".join(["lib", dirname, args.name, camel_name])
    save_to_json(var_file_path, var_data)

    modified_files = [init_file_path, var_file_path]
    created_files = [file_path]

    print("The following files have been created: {}".format(", ".join(created_files)))
    print("The following files have been modified: {}".format(", ".join(modified_files)))
