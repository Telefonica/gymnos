#
#
#   Train CLI app
#
#


import os
import uuid
import gymnos
import platform
import logging
import logging.config

from datetime import datetime

from ..utils.io_utils import save_to_json, read_json
from ..utils.platform_info import get_cpu_info, get_gpus_info, get_git_revision_hash


# MARK: Public methods

def add_arguments(parser):
    parser.add_argument("training_specs_json", help=("Training configuration JSON path. " +
                                                     "This will call Trainer.from_dict() with the JSON file."),
                        type=str)
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
    parser.add_argument("--environment", help="Execution environment to run experiment", type=str, default=None)
    parser.add_argument("--monitor", help="Whether or not monitor training from execution environment",
                        action="store_true", default=False)


def run_command(args):
    training_config = read_json(args.training_specs_json)

    trainer = gymnos.trainer.Trainer.from_dict(training_config)

    logger = logging.getLogger(__name__)

    if args.environment is not None:
        setup_basic_log_config()

        logger.info("Training will be executed in external environment")

        environment_instance = gymnos.execution_environments.load(args.environment)
        train_kwargs = environment_instance.train(trainer)

        logger.info("Training outputs: {}".format(train_kwargs))

        if args.monitor:
            logger.info("Monitoring will begin")
            environment_instance.monitor(**train_kwargs)

        return train_kwargs

    format_kwargs = dict(
        dataset_type=training_config["dataset"]["dataset"]["type"],
        model_type=training_config["model"]["model"]["type"],
        uuid=uuid.uuid4().hex,
        now=datetime.now()
    )

    execution_dir = args.execution_dir.format(**format_kwargs)
    trackings_dir = args.trackings_dir.format(**format_kwargs)

    os.makedirs(execution_dir)
    os.makedirs(trackings_dir, exist_ok=True)

    logging_config = read_json(os.path.join(os.path.dirname(__file__), "..", "config", "logging.json"))
    logging_config["handlers"]["file"]["filename"] = os.path.join(execution_dir,
                                                                  logging_config["handlers"]["file"]["filename"])
    logging.config.dictConfig(logging_config)

    logger.info("Execution directory will be located at {}".format(execution_dir))
    logger.info("Trackings directory will be located at {}".format(trackings_dir))

    dl_manager = gymnos.services.DownloadManager(args.download_dir, force_download=args.force_download,
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
        save_to_json(path, training_config)

    if not args.no_save_metrics:
        path = os.path.join(execution_dir, "metrics.json")
        logger.info("Saving metrics to {}".format(path))
        save_to_json(path, training_results)

    if not args.no_save_platform_info:
        path = os.path.join(execution_dir, "platform_info.json")
        logger.info("Saving platform info to {}".format(path))
        save_to_json(path, get_platform_info())


# MARK: Helpers

def setup_basic_log_config():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )


def get_platform_info():
    logger = logging.getLogger(__name__)

    info = dict(platform=platform.platform())

    try:
        info["cpu"] = get_cpu_info()
    except Exception:
        logger.exception("Error retrieving CPU information")

    try:
        info["gpu"] = get_gpus_info()
    except Exception:
        logger.exception("Error retrieving GPU information")

    try:
        info["gymnos"] = dict(git_hash=get_git_revision_hash())
    except Exception:
        logger.exception("Error retrieving git revision hash")

    return info
