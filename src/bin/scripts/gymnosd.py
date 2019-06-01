#!/usr/bin/python3

import os
import uuid
import copy
import logging
import argparse

from datetime import datetime
from tempfile import TemporaryDirectory

from lib.utils.path import chdir
from lib.datasets import HDF5Dataset
from lib.trainer import Trainer
from lib.core.model import Model
from lib.core.dataset import Dataset
from lib.core.training import Training
from lib.core.tracking import Tracking
from lib.core.experiment import Experiment
from lib.services.download_manager import DownloadManager
from lib.utils.io_utils import save_to_json, read_from_json

LOGGING_CONFIG_PATH = os.path.join("config", "logging.json")

DEFAULT_PREFERENCES_CONFIG_PATH = os.path.join("config", "preferences.json")
LOCAL_PREFERENCES_CONFIG_PATH = os.path.join("config", "preferences.local.json")

REGRESSION_TESTS_DIR = "experiments/tests"


def read_preferences():
    """
    Read default preferences and override properties defined by local preferences

    Returns
    -------
    config: dict
        Dictionnary with preferences
    """
    config = read_from_json(DEFAULT_PREFERENCES_CONFIG_PATH, with_comments_support=True)

    if os.path.isfile(LOCAL_PREFERENCES_CONFIG_PATH):
        local_preferences_config = read_from_json(LOCAL_PREFERENCES_CONFIG_PATH, with_comments_support=True)
        config.update(local_preferences_config)

    # Windows and Unix systems use different path separators, we normalize paths to adjust paths
    # separators to current system
    keys_to_normalize = ["download_dir", "extract_dir", "hdf5_datasets_dir", "executions_dir", "trackings_dir",
                         "trained_model_dir", "trained_pipeline_filename", "training_config_filename",
                         "execution_results_filename"]

    for key in keys_to_normalize:
        if config[key] is not None:
            config[key] = os.path.expanduser(os.path.normpath(config[key]))

    return config


def run_experiment(training_config_path):
    """
    Run trainer given training configuration

    Parameters
    ---------
    training_config_path: str
        Training configuration file path
    """
    config = read_preferences()

    training_config = read_from_json(training_config_path, with_comments_support=True)
    training_config_copy = copy.deepcopy(training_config)

    format_kwargs = {
        "dataset_name": training_config["dataset"]["name"],
        "model_name": training_config["model"]["name"],
        "uuid": uuid.uuid4().hex,
        "datetime": datetime.now()
    }

    download_dir = config["download_dir"].format(**format_kwargs)
    extract_dir = config["extract_dir"].format(**format_kwargs)
    executions_dir = config["executions_dir"].format(**format_kwargs)
    trackings_dir = config["trackings_dir"].format(**format_kwargs)
    default_experiment_name = config["default_experiment_name"].format(**format_kwargs)

    # if the user has not provided an experiment name, we set the experiment name
    # to the default experiment name from config
    training_config["experiment"] = training_config.get("experiment", {})
    if training_config["experiment"].get("name") is None:
        training_config["experiment"]["name"] = default_experiment_name

    execution_dir = os.path.join(executions_dir, training_config["experiment"]["name"])

    os.makedirs(execution_dir)
    os.makedirs(trackings_dir, exist_ok=True)

    logging_config = read_from_json(LOGGING_CONFIG_PATH)
    logging_config["handlers"]["file"]["filename"] = os.path.join(execution_dir,
                                                                  logging_config["handlers"]["file"]["filename"])
    logging.config.dictConfig(logging_config)

    logger = logging.getLogger(__name__)

    dataset = Dataset(**training_config["dataset"])
    model = Model(**training_config["model"])
    training = Training(**training_config.get("training", {}))  # optional field
    experiment = Experiment(**training_config.get("experiment", {}))  # optional field
    tracking = Tracking(**training_config.get("tracking", {}))  # optional field)

    hdf5_dataset_file_path = os.path.join(config["hdf5_datasets_dir"], dataset.name + config["hdf5_file_extension"])

    # Replace dataset by HDF5 dataset if HDF5 file exists
    if os.path.isfile(hdf5_dataset_file_path):
        dataset.dataset = HDF5Dataset(hdf5_dataset_file_path, features_key=config["hdf5_features_key"],
                                      labels_key=config["hdf5_labels_key"], info_key=config["hdf5_info_key"])

    dl_manager = DownloadManager(download_dir=download_dir, extract_dir=extract_dir,
                                 force_download=config["force_download"], force_extraction=config["force_extraction"])

    trainer = Trainer(dl_manager, trackings_dir=trackings_dir)

    success = False

    try:
        logger.info("Starting gymnos trainer ...")

        with chdir(execution_dir):
            results = trainer.train(experiment, model, dataset, training, tracking)

        success = True

        if config["save_execution_results"]:
            save_to_json(os.path.join(execution_dir, config["execution_results_filename"]), results)
    except Exception as e:
        logger.exception("Exception ocurred: {}".format(e))
        raise
    finally:
        if (success and config["save_trained_model"]) or (not success and config["save_trained_model_if_errors"]):
            save_model_dir = os.path.join(execution_dir, config["trained_model_dir"])
            os.makedirs(save_model_dir, exist_ok=True)
            model.model.save(save_model_dir)

        if (success and config["save_trained_pipeline"]) or (not success and config["save_trained_model_if_errors"]):
            dataset.pipeline.save(os.path.join(execution_dir, config["trained_pipeline_filename"]))

        if (success and config["save_training_config"]) or (not success and config["save_training_config_if_errors"]):
            save_to_json(os.path.join(execution_dir, config["training_config_filename"]), training_config_copy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-c", "--training_config", help="sets training training configuration file",
                       action="store")
    group.add_argument("-t", "--regression_test", help="execute regression test", action="store_true")
    args = parser.parse_args()

    if args.regression_test:
        test_config_filenames = os.listdir(REGRESSION_TESTS_DIR)
        with TemporaryDirectory() as temp_dir:
            for i, test_config_filename in enumerate(test_config_filenames):
                print("{}{} / {} - Current regression test: {}{}".format("\033[91m", i + 1, len(test_config_filenames),
                                                                         test_config_filename, "\033[0m"))
                run_experiment(os.path.join(REGRESSION_TESTS_DIR, test_config_filename))
    else:
        run_experiment(args.training_config)
