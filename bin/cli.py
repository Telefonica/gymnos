#
#
#   Command Line Interface
#
#

import os
import uuid
import argparse
import logging.config


from collections import OrderedDict

from .lib.io_utils import read_json, save_to_json

from gymnos.trainer import Trainer
from gymnos.services.download_manager import DownloadManager


def train(training_specs_json, download_dir, force_download, force_extraction, save_trainer, save_training_specs,
          save_results, execution_dir, trackings_dir):

    training_config = read_json(training_specs_json, object_pairs_hook=OrderedDict)

    trainer = Trainer.from_spec(training_config)

    dl_manager = DownloadManager(download_dir, force_download=force_download,
                                 force_extraction=force_extraction)

    format_kwargs = dict(dataset_name=trainer.dataset.name, model_name=trainer.model.name, uuid=uuid.uuid4().hex)

    execution_dir = execution_dir.format(**format_kwargs)
    trackings_dir = trackings_dir.format(**format_kwargs)

    os.makedirs(execution_dir)
    os.makedirs(trackings_dir, exist_ok=True)

    logging_config = read_json(os.path.join(os.path.dirname(__file__), "config", "logging.json"))
    logging_config["handlers"]["file"]["filename"] = os.path.join(execution_dir,
                                                                  logging_config["handlers"]["file"]["filename"])
    logging.config.dictConfig(logging_config)

    logger = logging.getLogger(__name__)

    try:
        training_results = trainer.train(dl_manager, trackings_dir=trackings_dir)
    except Exception as e:
        logger.exception("Exception ocurred: {}".format(e))
        raise

    if save_trainer:
        trainer.save(os.path.join(execution_dir, "saved_trainer.zip"))

    if save_training_specs:
        save_to_json(os.path.join(execution_dir, "training_specs.json"), training_config)

    if save_results:
        save_to_json(os.path.join(execution_dir, "results.json"), training_results)


def predict(samples_json, trainer):
    trainer = Trainer.load(trainer)

    samples = read_json(samples_json)

    predictions = trainer.predict(samples)

    print(predictions)


def main():
    parser = argparse.ArgumentParser(description="Gymnos tool")

    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("training_specs_json", help="Training configuration JSON path", type=str)

    train_parser.add_argument("--download_dir", help="Directory to download datasets", type=str, default="downloads")
    train_parser.add_argument("--force_download", help="Whether or not force download if file already exists",
                              action="store_true", default=False)
    train_parser.add_argument("--force_extraction", help="Whether or not force extraction if file already exists",
                              action="store_true", default=False)

    train_parser.add_argument("--save_trainer", help="Whether or not save trainer after training", action="store_true",
                              default=True)
    train_parser.add_argument("--save_training_specs", help="Whether or not save JSON training configuration file",
                              type=str, default=True)
    train_parser.add_argument("--save_results", help="Whether or not save results", type=str, default=True)

    train_parser.add_argument("--execution_dir", help="Execution directory to store training outputs", type=str,
                              default="trainings/{dataset_name}/executions/{uuid}")
    train_parser.add_argument("--trackings_dir", help="Trackings directory", type=str,
                              default="trainings/{dataset_name}/trackings")

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("samples_json", help="Samples to predict", type=str)
    predict_parser.add_argument("--trainer", help="Saved trainer file", type=str, required=True)

    args = parser.parse_args()

    if args.command == "train":
        train(args.training_specs_json, download_dir=args.download_dir, force_download=args.force_download,
              force_extraction=args.force_extraction, save_trainer=args.save_trainer,
              save_training_specs=args.save_training_specs, save_results=args.save_results,
              execution_dir=args.execution_dir, trackings_dir=args.trackings_dir)
    elif args.command == "predict":
        predict(args.samples_json, trainer=args.trainer)


main()
