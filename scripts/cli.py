#
#
#   Command Line Interface
#
#

import os
import uuid
import json
import argparse
import itertools
import numpy as np
import logging.config

from PIL import Image
from collections import OrderedDict


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def _json_default(o):
    # Hack to save numpy number with JSON module
    if isinstance(o, np.int32) or isinstance(o, np.int64):
        return int(o)
    if isinstance(o, np.float32) or isinstance(o, np.float64):
        return float(o)
    raise TypeError


def read_json(file_path, *args, **kwargs):
    """
    Read JSON

    Parameters
    ----------
    file_path: str
        JSON file path

    Returns
    -------
    json: dict
        JSON data.
    """
    with open(file_path) as f:
        return json.load(f, *args, **kwargs)


def save_to_json(path, obj, indent=4, *args, **kwargs):
    """
    Save data to JSON file.

    Parameters
    ----------
    path: str
        JSON file path.
    obj: dict or list
        Object to save
    indent: int, optional
        Indentation to save file (pretty print JSON)
    """
    with open(path, "w") as outfile:
        json.dump(obj, outfile, indent=indent, cls=NumpyEncoder, default=_json_default, *args, **kwargs)


def train(training_specs_json, download_dir, force_download, force_extraction, no_save_trainer, no_save_training_specs,
          no_save_results, execution_dir, trackings_dir):

    from gymnos.trainer import Trainer
    from gymnos.services.download_manager import DownloadManager

    training_config = read_json(training_specs_json, object_pairs_hook=OrderedDict)

    trainer = Trainer.from_dict(training_config)

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

    logger.info("Execution directory will be located at {}".format(execution_dir))
    logger.info("Trackings directory will be located at {}".format(trackings_dir))

    try:
        training_results = trainer.train(dl_manager, trackings_dir=trackings_dir)
    except Exception as e:
        logger.exception("Exception ocurred: {}".format(e))
        raise

    if not no_save_trainer:
        path = os.path.join(execution_dir, "saved_trainer.zip")
        logger.info("Saving trainer to {}".format(path))
        trainer.save(path)

    if not no_save_training_specs:
        path = os.path.join(execution_dir, "training_specs.json")
        logger.info("Saving training specs to {}".format(path))
        save_to_json(path, training_config)

    if not no_save_results:
        path = os.path.join(execution_dir, "results.json")
        logger.info("Saving results to {}".format(path))
        save_to_json(path, training_results)


def predict(saved_trainer, json_file=None, images=None):
    from gymnos.trainer import Trainer
    from gymnos.datasets.dataset import ClassLabel

    if images is None:
        samples = np.array(read_json(json_file))
    else:
        # images can be a nested list so we flatten the list
        samples = np.array([np.array(Image.open(image).convert("RGB")) for image in itertools.chain(*images)])

    trainer = Trainer.load(saved_trainer)

    response = dict(predictions=trainer.predict(samples))

    try:
        response["probabilities"] = trainer.predict_proba(samples)
    except NotImplementedError:
        pass

    labels = trainer.dataset.dataset.info().labels
    if isinstance(labels, ClassLabel):
        response["classes"] = dict(
            names=labels.names,
            total=labels.num_classes
        )

    print(json.dumps(response, cls=NumpyEncoder, indent=4))


def serve(saved_trainer, host=None, port=None, debug=None):
    import flask

    from gymnos.trainer import Trainer
    from gymnos.datasets.dataset import ClassLabel

    class FlaskNumpyEncoder(flask.json.JSONEncoder):
        """
        Flask Json Encoder to handle Numpy arrays
        """

        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    app = flask.Flask(__name__)
    app.json_encoder = FlaskNumpyEncoder

    @app.route("/", methods=["GET"])
    def info():
        # FIXME: the trainer should be loaded before every request but there are issues combining tf with flask requests
        trainer = Trainer.load(saved_trainer)

        return flask.jsonify(trainer.to_dict())

    @app.route("/", methods=["POST"])
    def predict():
        # FIXME: same as above
        trainer = Trainer.load(saved_trainer)

        response = dict(predictions=trainer.predict(flask.request.json))

        try:
            response["probabilities"] = trainer.predict_proba(flask.request.json)
        except NotImplementedError:
            pass

        labels = trainer.dataset.dataset.info().labels
        if isinstance(labels, ClassLabel):
            response["classes"] = dict(
                names=labels.names,
                total=labels.num_classes
            )

        return flask.jsonify(response)

    app.run(host=host, port=port, debug=debug)


def build_parser():
    parser = argparse.ArgumentParser(description="Gymnos tool")

    subparsers = parser.add_subparsers(dest="command")

    # MARK: Train parsers

    train_parser = subparsers.add_parser("train", help="Train gymnos trainer using a JSON file")
    train_parser.add_argument("training_specs_json", help=("Training configuration JSON path. " +
                                                           "This will call Trainer.from_dict() with the JSON file."),
                              type=str)
    train_parser.add_argument("--download_dir", help="Directory to download datasets", type=str, default="downloads")
    train_parser.add_argument("--force_download", help="Whether or not force download if file already exists",
                              action="store_true", default=False)
    train_parser.add_argument("--force_extraction", help="Whether or not force extraction if file already exists",
                              action="store_true", default=False)
    train_parser.add_argument("--no-save_trainer", help="Whether or not save trainer after training",
                              action="store_true", default=False)
    train_parser.add_argument("--no-save_training_specs", help="Whether or not save JSON training configuration file",
                              action="store_true", default=False)
    train_parser.add_argument("--no-save_results", help="Whether or not save results", action="store_true",
                              default=False)
    train_parser.add_argument("--execution_dir", help="Execution directory to store training outputs. It accepts the" +
                                                      " following format arguments: dataset_name, model_name, uuid",
                              type=str, default="trainings/{dataset_name}/executions/{uuid}")
    train_parser.add_argument("--trackings_dir", help="Execution directory to store tracking outputs. It accepts the" +
                                                      " following format arguments: dataset_name, model_name, uuid",
                              type=str, default="trainings/{dataset_name}/trackings")

    # MARK: Predict parsers

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("saved_trainer", help="Saved trainer file", type=str)

    predict_parser_group = predict_parser.add_mutually_exclusive_group(required=True)
    predict_parser_group.add_argument("--json", help="JSON file path with list of samples to predict", type=str)
    predict_parser_group.add_argument("--image", help="Image file path to predict", nargs="*", type=str,
                                      action="append")

    # MARK: Serve parsers

    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("saved_trainer", help="Saved trainer file", type=str)
    serve_parser.add_argument("--host", help="Hostname to listen to.", type=str, required=False)
    serve_parser.add_argument("--port", help="Port to listen to.", type=int, required=False)
    serve_parser.add_argument("--debug", help="Enable or disable debug mode", default=False, required=False,
                              action="store_true")

    return parser


def main():
    parser = build_parser()

    args = parser.parse_args()

    if args.command == "train":
        train(args.training_specs_json, download_dir=args.download_dir, force_download=args.force_download,
              force_extraction=args.force_extraction, no_save_trainer=args.no_save_trainer,
              no_save_training_specs=args.no_save_training_specs, no_save_results=args.no_save_results,
              execution_dir=args.execution_dir, trackings_dir=args.trackings_dir)
    elif args.command == "predict":
        predict(args.saved_trainer, args.json, args.image)
    elif args.command == "serve":
        serve(args.saved_trainer, host=args.host, port=args.port, debug=args.debug)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
