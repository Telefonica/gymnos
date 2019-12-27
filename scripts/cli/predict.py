#
#
#   Predict CLI app
#
#

import json
import itertools
import numpy as np

from ..utils.io_utils import read_json

from gymnos.trainer import Trainer
from gymnos.datasets.dataset import ClassLabel
from gymnos.utils.image_utils import imread_rgb
from gymnos.utils.json_utils import NumpyEncoder


def add_arguments(parser):
    parser.add_argument("saved_trainer", help="Saved trainer file", type=str)

    parser_group = parser.add_mutually_exclusive_group(required=True)
    parser_group.add_argument("--json", help="JSON file path with list of samples to predict", type=str)
    parser_group.add_argument("--image", help="Image file path to predict", nargs="*", type=str,
                              action="append")


def run_command(args):
    if args.image is None:
        samples = np.array(read_json(args.json))
    else:
        # images can be a nested list so we flatten the list
        samples = np.array([np.array(imread_rgb(image)) for image in itertools.chain(*args.image)])

    trainer = Trainer.load(args.saved_trainer)

    response = dict(predictions=trainer.predict(samples))

    try:
        response["probabilities"] = trainer.predict_proba(samples)
    except NotImplementedError:
        pass

    labels = trainer.dataset.dataset.labels_info
    if isinstance(labels, ClassLabel):
        response["classes"] = dict(
            names=labels.names,
            total=labels.num_classes
        )

    print(json.dumps(response, cls=NumpyEncoder, indent=4))
