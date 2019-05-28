#!/usr/bin/python3

import argparse
import copy
import logging
import os
import shutil
from tempfile import TemporaryDirectory

from lib.core.dataset import Dataset
from lib.core.experiment import Experiment
from lib.core.model import Model
from lib.core.tracking import Tracking
from lib.core.training import Training
from lib.logger import get_logger
from lib.predictor import Predictor
from lib.trainer import Trainer
from lib.utils.io_utils import save_to_json, read_from_json

CACHE_CONFIG_PATH = os.path.join("config", "cache.json")
LOGGING_CONFIG_PATH = os.path.join("config", "logging.json")

REGRESSION_TESTS_DIR = os.path.join("experiments", "tests")

TRAINING_LOG_FILENAME = "execution.log"
TRAINING_CONFIG_FILENAME = "training_config.json"
ARTIFACTS_PATH = "artifacts"


def run_experiment(training_config_path, output_path="trainings"):
    training_config = read_from_json(training_config_path)
    training_config_copy = copy.deepcopy(training_config)

    cache_config = read_from_json(CACHE_CONFIG_PATH)
    logging_config = read_from_json(LOGGING_CONFIG_PATH)
    logging.config.dictConfig(logging_config)
    logger = get_logger(prefix="Main")

    logger.info("Starting gymnos environment ...")

    os.makedirs(cache_config["datasets"], exist_ok=True)

    trainer = Trainer(trainings_path=output_path, cache_datasets_path=cache_config["datasets"])

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


def run_prediction(execution_dir, values, loggging_config_path_prediction):
    prediction_config = read_from_json(os.path.join(execution_dir, "training_config.json"))

    logging_config = read_from_json(loggging_config_path_prediction)
    logging.config.dictConfig(logging_config)
    logger = get_logger(prefix="Main")

    model = Model(**prediction_config["model"])
    dataset = Dataset(**prediction_config["dataset"])

    artifacts_path = os.path.join(execution_dir, ARTIFACTS_PATH)
    model.model.restore(artifacts_path)
    dataset.preprocessor_pipeline.restore(os.path.join(artifacts_path, "pipeline.pkl"))

    predictor = Predictor(model, dataset)

    try:
        prediction = predictor.predict(values)

    except Exception as e:
        logger.exception("Exception ocurred: {}".format(e))
        raise
    return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-c", "--training_config", help="sets training training configuration file",
                       action="store")
    group.add_argument("-t", "--regression_test", help="execute regression test", action="store_true")

    args = parser.parse_args()

    if args.regression_test:
        test_config_filenames = os.listdir(REGRESSION_TESTS_DIR)
        for i, test_config_filename in enumerate(test_config_filenames):
            print("{}{} / {} - Current regression test: {}{}".format("\033[91m", i + 1, len(test_config_filenames),
                                                                     test_config_filename, "\033[0m"))
            with TemporaryDirectory() as temp_dir:
                run_experiment(os.path.join(REGRESSION_TESTS_DIR, test_config_filename), temp_dir)
    else:
        run_experiment(args.training_config)
