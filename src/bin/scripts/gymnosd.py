#!/usr/bin/python3

import os
import copy
import shutil
import logging
import argparse
import traceback

from lib.logger import get_logger
from lib.trainer import Trainer
from lib.core.model import Model
from lib.core.dataset import Dataset
from lib.core.training import Training
from lib.core.session import Session
from lib.core.tracking import Tracking
from lib.core.experiment import Experiment
from lib.utils.io_utils import save_to_json, read_from_json

CACHE_CONFIG_PATH = os.path.join("config", "cache.json")
LOGGING_CONFIG_PATH = os.path.join("config", "logging.json")

TRAINING_LOG_FILENAME = "execution.log"
TRAINING_CONFIG_FILENAME = "training_config.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--training_config", help="sets training training configuration file path",
                        action="store", required=True)
    args = parser.parse_args()

    training_config = read_from_json(args.training_config)
    training_config_copy = copy.deepcopy(training_config)

    cache_config = read_from_json(CACHE_CONFIG_PATH)
    logging_config = read_from_json(LOGGING_CONFIG_PATH)

    logging.config.dictConfig(logging_config)

    logger = get_logger(prefix="Main")

    logger.info("Starting gymnos environment ...")

    os.makedirs(cache_config["datasets"], exist_ok=True)

    trainer = Trainer(
        model=Model(**training_config["model"]),
        dataset=Dataset(cache_dir=cache_config["datasets"], **training_config["dataset"]),
        training=Training(**training_config.get("training", {})),  # optional field
        experiment=Experiment(**training_config.get("experiment", {})),  # optional field
        tracking=Tracking(**training_config.get("tracking", {})),  # optional field
        session=Session(**training_config.get("session", {}))  # optional field
    )

    try:
        execution_path = trainer.run()

        # save original config to execution path
        save_to_json(os.path.join(execution_path, TRAINING_CONFIG_FILENAME), training_config_copy)
        # save logs to execution path
        shutil.move(logging_config["handlers"]["file_handler"]["filename"], os.path.join(execution_path,
                                                                                         TRAINING_LOG_FILENAME))

        logger.info("Success! Execution saved ({})".format(execution_path))
    except Exception as e:
        logger.error("Exception ocurred: {}".format(e))
        traceback.print_exc()
