#
#
#   Create new model/dataset/preprocessor/tracker/data_augmentor
#
#

import os
import re
import gymnos
import argparse
from inspect import cleandoc

SNAKE_CASE_TEST_RE = re.compile(r'^[a-z]+([a-z\d]+_|_[a-z\d]+|[a-z\d]+)+[a-z\d]+$')
SNAKE_CASE_TEST_DASH_RE = re.compile(r'^[a-z]+([a-z\d]+-|-[a-z\d]+)+[a-z\d]+$')
SNAKE_CASE_REPLACE_RE = re.compile(r'(_)([a-z\d])')
SNAKE_CASE_REPLACE_DASH_RE = re.compile(r'(-)([a-z\d])')


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
        re_template = r"^[a-z]+([a-z\d]+{sign}|{sign}[a-z\d]+|[a-z\d]+)+[a-z\d]+$"
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
    r = re_map.get(separator, re.compile(r'({sign})([a-z\d])'.format(sign=re.escape(separator))))
    string = r.sub(lambda m: m.group(2).upper(), string)
    if upper_case_first:
        return string[0].upper() + string[1:]
    return string


DATASET_TYPE = "dataset"
MODEL_TYPE = "model"
PREPROCESSOR_TYPE = "preprocessor"
TRACKER_TYPE = "tracker"
DATA_AUGMENTOR_TYPE = "data_augmentor"
SERVICE_TYPE = "service"
EXECUTION_ENVIRONMENT_TYPE = "execution_environment"

DATASET_FILE_STR = """
#
#
#   {name}
#
#

from .dataset import Dataset, Array, ClassLabel


class {name}:
    \"""
    {TODO}: Description of my dataset.
    \"""

    @property
    def features_info(self):
        # {TODO}: Specifies the information about the features (shape, dtype, etc...)

    @property
    def labels_info(self):
        # {TODO}: Specifies the information about the labels (shape, dtype, etc ...)

    def download_and_prepare(self, dl_manager):
        pass  # {TODO}: download any file you will need later in the __getitem__ and __len__ function

    def __getitem__(self, given):
        pass  # {TODO}: Get dataset item/s. Given can be a slice object or an int. Called after download_and_prepare.

    def __len__(self):
        pass  # {TODO}: Dataset length. Called after download_and_prepare
"""

MODEL_FILE_STR = """
#
#
#   {name}
#
#

from .model import Model


class {name}(Model):
    \"""
    {TODO}: Description of my model.
    \"""

    def __init__(self, **parameters):
        pass  # {TODO}: Define and initialize model parameters.

    def fit(self, X, y, **training_parameters):
        pass  # {TODO}: Fit model to training data.

    def fit_generator(self, X, y, **training_parameters):
        # {OPTIONAL}: Fit model to training generator. Write method if your model supports incremental learning
        raise NotImplementedError()

    def predict(self, X):
        pass  # {TODO}: Predict classes/values using features.

    def predict_proba(self, X):
        # {OPTIONAL}: Predict probabilities using features. Write method if your model is a probabilistic model
        raise NotImplementedError()

    def evaluate(self, X, y):
        pass  # {TODO}: Evaluate model.

    def save(self, save_dir):
        pass  # {TODO}: Save model to save_dir.

    def restore(self, save_dir):
        pass  # {TODO}: Restore model from save_dir.
"""

PREPROCESSOR_FILE_STR = """
#
#
#   {name}
#
#

from .preprocessor import Preprocessor


class {name}(Preprocessor):
    \"""
    {TODO}: Description of my preprocessor.
    \"""

    def __init__(self, **parameters):
        pass  # {TODO}: Define and initialize model parameters

    def fit(self, X, y=None):
        pass  # {TODO}: Fit preprocessor to training data.

    def fit_generator(self, generator):
        # {OPTIONAL}: Fit preprocessor to training generator. Only if your preprocessor supports incremental learning
        raise NotImplementedError()

    def transform(self, X):
        pass  # {TODO}: Preprocess data
"""

TRACKER_FILE_STR = """
#
#
#   {name}
#
#

from .tracker import Tracker


class {name}(Tracker):
    \"""
    {TODO}: Description of my tracker
    \"""

    def __init__(self, **parameters):
        pass  # {TODO}: Define and initialize tracker parameters

    def start(self, run_id, logdir):
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

    def end(self):
        # {OPTIONAL}: Called when the experiment is finished
        pass
"""

DATA_AUGMENTOR_FILE_STR = """
#
#
#   {name}
#
#

from .data_augmentor import DataAugmentor


class {name}(DataAugmentor):
    \"""
    {TODO}: Description of my preprocessor.
    \"""

    def __init__(self, probability, **parameters):
        super().__init__(probability)
        pass  # {TODO}: Define and initialize model parameters

    def fit(self, X, y=None):
        pass  # {TODO}: Fit preprocessor to training data.

    def fit_generator(self, generator):
        # {OPTIONAL}: Fit preprocessor to training generator. Only if your preprocessor supports incremental learning
        raise NotImplementedError()

    def transform(self, X):
        pass  # {TODO}: Preprocess data
"""

SERVICE_FILE_STR = """
#
#
#   {name}
#
#

from .. import config
from .service import Service


class {name}(Service):
    \"""
    {TODO}: Description of my service.
    \"""

    class Config(config.Config):
        pass  # {OPTIONAL}: Define your required and optional configuration variables.

    def download(self, *args, **kwargs):
        pass  # {OPTIONAL}: Download file.
"""

EXECUTION_ENVIRONMENT_FILE_STR = """
#
#
#   {name}
#
#

from .. import config
from .execution_environment import ExecutionEnvironment


class {name}(ExecutionEnvironment):
    \"""
    {TODO}: Description of my execution environment.
    \"""

    class Config(config.Config):
        pass  # {OPTIONAL}: Define your required and optional configuration variables.

    def train(self):
        pass  # {OPTIONAL}: Train experiment with execution environment

    def monitor(self, **train_kwargs):
        pass  # {OPTIONAL}: Monitor training with kwargs from train() as arguments
"""


def create_component(raw_string, name, dirname, registry):
    if name in registry:
        raise ValueError("Component with name {} already exists".format(name))

    camel_name = snake_case_to_camel(name)
    file_str = raw_string.format(name=camel_name, TODO="TODO({})".format(camel_name),
                                 OPTIONAL="OPTIONAL({})".format(camel_name))

    file_dir = os.path.join("gymnos", dirname)
    file_path = os.path.join(file_dir, name + ".py")

    with open(file_path, "x") as archive:
        archive.write(file_str)

    init_file_path = os.path.join("gymnos", "__init__.py")

    with open(init_file_path, "a") as fp:
        fp.write("\n")
        fp.write(cleandoc("""
            {}.register(
                name="{}",
                entry_point="{}"
            )
        """.format(dirname, name, ".".join(["gymnos", dirname, name, camel_name]))))
        fp.write("\n")

    return [init_file_path, file_path]


def create_dataset(dataset_name):
    return create_component(DATASET_FILE_STR, dataset_name, "datasets", gymnos.datasets.registry)


def create_model(model_name):
    return create_component(MODEL_FILE_STR, model_name, "models", gymnos.models.registry)


def create_tracker(tracker_name):
    return create_component(TRACKER_FILE_STR, tracker_name, "trackers", gymnos.trackers.registry)


def create_preprocessor(preprocessor_name):
    return create_component(PREPROCESSOR_FILE_STR, preprocessor_name, "preprocessors",
                            gymnos.preprocessors.registry)


def create_data_augmentor(data_augmentor_name):
    return create_component(DATA_AUGMENTOR_FILE_STR, data_augmentor_name, "data_augmentors",
                            gymnos.data_augmentors.registry)


def create_service(service_name):
    return create_component(SERVICE_FILE_STR, service_name, "services",
                            gymnos.services.registry)


def create_execution_environment(execution_environment_name):
    return create_component(EXECUTION_ENVIRONMENT_FILE_STR, execution_environment_name, "execution_environments",
                            gymnos.execution_environments.registry)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("type", type=str, choices=[DATASET_TYPE, MODEL_TYPE, PREPROCESSOR_TYPE, TRACKER_TYPE,
                                                   DATA_AUGMENTOR_TYPE, SERVICE_TYPE, EXECUTION_ENVIRONMENT_TYPE])
    parser.add_argument("-n", "--name", type=str, required=True, help="Name to generate files (snake case)")
    args = parser.parse_args()

    if not is_snake_case(args.name):
        error_str = ("Name must be snake case. "
                     "A string is considered snake case when: \n" +
                     "    * it's composed only by lowercase letters ([a-z]), underscores " +
                     " and optionally numbers ([0-9])\n" +
                     "    * it does not start/end with an underscore\n" +
                     "    * it does not start with a number\n")
        raise ValueError(error_str)

    if args.type == DATASET_TYPE:
        modified_files = create_dataset(args.name)
    elif args.type == MODEL_TYPE:
        modified_files = create_model(args.name)
    elif args.type == PREPROCESSOR_TYPE:
        modified_files = create_preprocessor(args.name)
    elif args.type == TRACKER_TYPE:
        modified_files = create_tracker(args.name)
    elif args.type == DATA_AUGMENTOR_TYPE:
        modified_files = create_data_augmentor(args.name)
    elif args.type == SERVICE_TYPE:
        modified_files = create_service(args.name)
    elif args.type == EXECUTION_ENVIRONMENT_TYPE:
        modified_files = create_execution_environment(args.name)
    else:
        parser.print_help()

    print("The following files have been modified: {}".format(", ".join(modified_files)))
