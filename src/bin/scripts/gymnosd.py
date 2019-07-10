#!/usr/bin/python3

import os
import uuid
import copy
import logging
import argparse

from glob import glob
from datetime import datetime

from lib.datasets import HDF5Dataset
from lib.trainer import Trainer
from lib.utils.termcolor import cprint
from lib.core.model import Model
from lib.core.dataset import Dataset
from lib.core.training import Training
from lib.core.tracking import Tracking
from lib.services.download_manager import DownloadManager
from lib.utils.io_utils import save_to_json, read_from_json

LOGGING_CONFIG_PATH = os.path.join("config", "logging.json")

DEFAULT_PREFERENCES_CONFIG_PATH = os.path.join("config", "preferences.json")
LOCAL_PREFERENCES_CONFIG_PATH = os.path.join("config", "preferences.local.json")


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
                         "trained_model_dir", "trained_preprocessors_filename", "training_config_filename",
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

    trackings_dir = os.path.abspath(trackings_dir)

    dataset = Dataset(**training_config["dataset"])
    model = Model(**training_config["model"])
    training = Training(**training_config.get("training", {}))  # optional field
    tracking = Tracking(**training_config.get("tracking", {}))  # optional field)

    execution_dir = os.path.join(executions_dir, tracking.run_id)

    os.makedirs(execution_dir)
    os.makedirs(trackings_dir, exist_ok=True)

    logging_config = read_from_json(LOGGING_CONFIG_PATH)
    logging_config["handlers"]["file"]["filename"] = os.path.join(execution_dir,
                                                                  logging_config["handlers"]["file"]["filename"])
    logging.config.dictConfig(logging_config)

    logger = logging.getLogger(__name__)

    logger.info("Starting gymnos trainer ...")

    hdf5_dataset_file_path = os.path.join(config["hdf5_datasets_dir"], dataset.name + config["hdf5_file_extension"])

    # Replace dataset by HDF5 dataset if HDF5 file exists
    if os.path.isfile(hdf5_dataset_file_path):
        logger.info("HDF5 dataset found. It will be used for training")
        dataset.dataset = HDF5Dataset(hdf5_dataset_file_path, features_key=config["hdf5_features_key"],
                                      labels_key=config["hdf5_labels_key"], info_key=config["hdf5_info_key"])

        logger.debug(('HDF5 "{}" key will be used to retrieve info, HDF5 "{}" key will be used to retrieve features ' +
                      'and HDF5 "{}" key will be used to retrieve labels').format(config["hdf5_info_key"],
                                                                                  config["hdf5_features_key"],
                                                                                  config["hdf5_labels_key"]))

    logger.debug("Downloads for dataset files will be located at {}".format(download_dir))
    logger.debug("Extractions for dataset files will be located at {}".format(extract_dir))
    dl_manager = DownloadManager(download_dir=download_dir, extract_dir=extract_dir,
                                 force_download=config["force_download"], force_extraction=config["force_extraction"])

    logger.debug("Trackings will be located at {}".format(trackings_dir))
    trainer = Trainer(dl_manager, trackings_dir=trackings_dir)

    success = False

    try:
        results = trainer.train(model, dataset, training, tracking)

        success = True
        logger.info("Execution succeed!")

        if config["save_execution_results"]:
            execution_results_path = os.path.join(execution_dir, config["execution_results_filename"])
            logger.info(("Saving execution results (elapsed times, metrics, system info, " +
                         "etc ...) to {}").format(execution_results_path))
            save_to_json(execution_results_path, results)
    except Exception as e:
        logger.exception("Exception ocurred: {}".format(e))
        raise
    finally:
        if (success and config["save_trained_model"]) or (not success and config["save_trained_model_if_errors"]):
            save_model_dir = os.path.join(execution_dir, config["trained_model_dir"])
            logger.info("Saving model to directory {}".format(save_model_dir))
            os.makedirs(save_model_dir, exist_ok=True)
            model.model.save(save_model_dir)

        if (success and config["save_trained_preprocessors"]) or (not success and
                                                                  config["save_trained_preprocessors_if_errors"]):
            pipeline_path = os.path.join(execution_dir, config["trained_preprocessors_filename"])
            logger.info("Saving preprocessors to {}".format(pipeline_path))
            dataset.preprocessors.save(pipeline_path)

        if (success and config["save_training_config"]) or (not success and config["save_training_config_if_errors"]):
            training_config_copy_path = os.path.join(execution_dir, config["training_config_filename"])
            logger.info("Saving original training configuration to {}".format(training_config_copy_path))
            save_to_json(training_config_copy_path, training_config_copy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-c", "--training_config", action="store",
                       help="Training config JSON file or directory with JSON training config files")
    args = parser.parse_args()

    if os.path.isdir(args.training_config):
        training_config_files = glob(os.path.join(args.training_config, "*.json"))
        cprint("Directory found with {} training files".format(len(training_config_files)), on_color="on_green")
        for index, training_config in enumerate(training_config_files):
            cprint("{}/{} - Executing {}".format(index + 1, len(training_config_files), training_config),
                   on_color='on_cyan')
            run_experiment(training_config)
    else:
        run_experiment(args.training_config)
