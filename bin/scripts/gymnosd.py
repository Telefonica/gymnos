#!/usr/bin/python3

import os
import json
import logging
import argparse
import traceback

from uuid import uuid4

from lib.logger import logger
from lib.trainer import Trainer
from lib.core.model import Model
from lib.core.dataset import Dataset
from lib.core.training import Training
from lib.core.session import Session
from lib.core.tracking import Tracking
from lib.core.experiment import Experiment

CACHE_CONFIG_PATH = os.path.join("config", "cache.json")
LOGGING_CONFIG_PATH = os.path.join("config", "logging.json")


def setup_logging():
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--training_config", help="sets training training configuration file path",
                        action="store", required=True)
    args = parser.parse_args()

    with open(args.training_config) as f:
        training_config = json.load(f)

    with open(CACHE_CONFIG_PATH) as f:
        cache_config = json.load(f)

    with open(LOGGING_CONFIG_PATH) as f:
        logging_config = json.load(f)

    setup_logging()

    logger.info("-" * 10 + " GYMNOS ENVIRONMENT STARTED " + "-" * 10)

    os.makedirs(cache_config["datasets"], exist_ok=True)

    trainer = Trainer(
        experiment=Experiment(id=uuid4(), **training_config["experiment"]),
        model=Model(**training_config["model"]),
        dataset=Dataset(cache=cache_config["datasets"], **training_config["dataset"]),
        training=Training(**training_config["training"]),
        tracking=Tracking(**training_config.get("tracking", {})),  # optional field
        session=Session(**training_config.get("session", {}))  # optional field
    )

    try:
        trainer.run()
    except Exception as e:
        logger.error("Exception ocurred: {}".format(e))
        traceback.print_exc()
