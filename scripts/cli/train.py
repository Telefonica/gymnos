#
#
#   Train CLI app
#
#

import os
import uuid
import pydoc
import gymnos
import logging
import argparse
import logging.config

from datetime import datetime
from gymnos.services import DownloadManager
from ..utils.platform_info import get_platform_info
from gymnos.utils.json_utils import read_json, save_to_json


logger = logging.getLogger(__name__)


# MARK: Public methods

def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("training_specs", help=("Training configuration JSON path. " +
                                                "This will call Trainer.from_dict() with the JSON file."),
                        type=str)
    subparser = parser.add_subparsers(dest="environment", help="Execution environment to run experiment",
                                      required=False)

    for execution_environment_spec in gymnos.execution_environments.registry.all():
        execution_environment_parser = subparser.add_parser(execution_environment_spec.type)
        execution_environment_cls = pydoc.locate(execution_environment_spec.entry_point)
        execution_environment_cls.add_arguments(execution_environment_parser)


def get_default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_dir", help="Directory to download datasets", type=str, default="downloads")
    parser.add_argument("--force_download", help="Whether or not force download if file already exists",
                        action="store_true", default=False)
    parser.add_argument("--force_extraction", help="Whether or not force extraction if file already exists",
                        action="store_true", default=False)
    parser.add_argument("--no-save_trainer", help="Whether or not save trainer after training",
                        action="store_true", default=False)
    parser.add_argument("--no-save_training_specs", help="Whether or not save JSON training configuration file",
                        action="store_true", default=False)
    parser.add_argument("--no-save_metrics", help="Whether or not save results", action="store_true",
                        default=False)
    parser.add_argument("--no-save_platform_info", help="Whether or not save current platform information",
                        action="store_true", default=False)
    parser.add_argument("--execution_dir", help="Execution directory to store training outputs. It accepts the " +
                                                "following format arguments: dataset_type, model_type, uuid, now",
                        type=str, default="trainings/{dataset_type}/executions/{now:%Y-%m-%d_%H-%M-%S}")
    parser.add_argument("--trackings_dir", help="Execution directory to store tracking outputs. It accepts the" +
                                                " following format arguments: dataset_type, model_type, uuid",
                        type=str, default="trainings/{dataset_type}/trackings")
    return parser


def parse_args(parser: argparse.ArgumentParser):
    args, extras = parser.parse_known_args()

    if args.environment is None:
        default_parser = get_default_parser()
        extras_args = default_parser.parse_args(extras)
        args = argparse.Namespace(**vars(args), **vars(extras_args))

    return args


def run_command(args: argparse.Namespace):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    training_config = read_json(args.training_specs)

    trainer = gymnos.Trainer.from_dict(training_config)

    if args.environment is None:
        _run_without_environment(trainer, args)
    else:
        _run_with_environment(trainer, args)


def _run_with_environment(trainer, args):
    execution_environment = gymnos.execution_environments.load(args.environment)

    execution_environment.train(trainer, **vars(args))


def _run_without_environment(trainer, args):
    format_kwargs = dict(
        dataset_type=trainer.dataset.dataset_spec.get("type", ""),
        model_type=trainer.model.model_spec.get("type", ""),
        uuid=uuid.uuid4().hex,
        now=datetime.now()
    )

    execution_dir = args.execution_dir.format(**format_kwargs)
    trackings_dir = args.trackings_dir.format(**format_kwargs)

    os.makedirs(execution_dir)
    os.makedirs(trackings_dir, exist_ok=True)

    logging_config = read_json(
        os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "config", "logging.json"))  # FIXME
    logging_config["handlers"]["file"]["filename"] = os.path.join(execution_dir,
                                                                  logging_config["handlers"]["file"]["filename"])
    logging.config.dictConfig(logging_config)

    logger.info("Execution directory will be located at {}".format(execution_dir))
    logger.info("Trackings directory will be located at {}".format(trackings_dir))

    dl_manager = DownloadManager(args.download_dir, force_download=args.force_download,
                                 force_extraction=args.force_extraction)

    try:
        training_results = trainer.train(dl_manager, trackings_dir=trackings_dir)
    except Exception as e:
        logger.exception("Exception ocurred: {}".format(e))
        raise

    if not args.no_save_trainer:
        path = os.path.join(execution_dir, "saved_trainer.zip")
        logger.info("Saving trainer to {}".format(path))
        trainer.save(path)

    if not args.no_save_training_specs:
        path = os.path.join(execution_dir, "training_specs.json")
        logger.info("Saving training specs to {}".format(path))
        save_to_json(path, trainer.to_dict())

    if not args.no_save_metrics:
        path = os.path.join(execution_dir, "metrics.json")
        logger.info("Saving metrics to {}".format(path))
        save_to_json(path, training_results)

    if not args.no_save_platform_info:
        path = os.path.join(execution_dir, "platform_info.json")
        logger.info("Saving platform info to {}".format(path))
        save_to_json(path, get_platform_info())
