#!/usr/bin/python3

import os
import copy
import shutil
import logging
import argparse

from tempfile import TemporaryDirectory

from lib.logger import get_logger
from lib.trainer import Trainer
from lib.core.model import Model
from lib.core.dataset import Dataset
from lib.core.training import Training
from lib.core.tracking import Tracking
from lib.core.experiment import Experiment
from lib.utils.io_utils import save_to_json, read_from_json

CACHE_CONFIG_PATH = os.path.join("config", "cache.json")
LOGGING_CONFIG_PATH = os.path.join("config", "logging.json")

REGRESSION_TESTS_DIR = os.path.join("experiments", "tests")

TRAINING_LOG_FILENAME = "execution.log"
TRAINING_CONFIG_FILENAME = "training_config.json"


def run_experiment(trainer, training_config_path):
    training_config = read_from_json(training_config_path)
    training_config_copy = copy.deepcopy(training_config)

    logging_config = read_from_json(LOGGING_CONFIG_PATH)
    logging.config.dictConfig(logging_config)
    logger = get_logger(prefix="Main")

    logger.info("Starting gymnos environment ...")

    os.makedirs(cache_config["datasets"], exist_ok=True)

    try:
        trainer.train(
            model=Model(**training_config["model"]),
            dataset=Dataset(**training_config["dataset"]),
            training=Training(**training_config.get("training", {})),  # optional field
            experiment=Experiment(**training_config.get("experiment", {})),  # optional field
            tracking=Tracking(**training_config.get("tracking", {}))  # optional field)
        )

        logger.info("Success! Execution saved ({})".format(trainer.last_execution_path_))
    except Exception as e:
        logger.exception("Exception ocurred: {}".format(e))
        raise
    finally:
        # save original config to execution path
        save_to_json(os.path.join(trainer.last_execution_path_, TRAINING_CONFIG_FILENAME), training_config_copy)
        # save logs to execution path
        shutil.move(logging_config["handlers"]["file_handler"]["filename"], os.path.join(trainer.last_execution_path_,
                                                                                         TRAINING_LOG_FILENAME))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-c", "--training_config", help="sets training training configuration file",
                       action="store")
    group.add_argument("-t", "--regression_test", help="execute regression test", action="store_true")
    args = parser.parse_args()

    cache_config = read_from_json(CACHE_CONFIG_PATH)

    if args.regression_test:
        test_config_filenames = os.listdir(REGRESSION_TESTS_DIR)
        with TemporaryDirectory() as temp_dir:
            trainer = Trainer(trainings_path=temp_dir, optimized_datasets_dir=cache_config["datasets"])
            for i, test_config_filename in enumerate(test_config_filenames):
                print("{}{} / {} - Current regression test: {}{}".format("\033[91m", i + 1, len(test_config_filenames),
                                                                         test_config_filename, "\033[0m"))
                run_experiment(trainer, os.path.join(REGRESSION_TESTS_DIR, test_config_filename))
    else:
        trainer = Trainer(optimized_datasets_dir=cache_config["datasets"])
        run_experiment(trainer, args.training_config)
