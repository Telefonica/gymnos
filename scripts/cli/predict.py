#
#
#   Predict CLI app
#
#

import json
import itertools
import numpy as np
import pandas as pd

from ..utils.io_utils import read_json

from gymnos.trainer import Trainer
from gymnos.datasets.dataset import ClassLabel
from gymnos.preprocessors.utils.image_ops import imread_rgb
from gymnos.utils.json_utils import NumpyEncoder


def add_arguments(parser):
    parser.add_argument("saved_trainer", help="Saved trainer file", type=str)

    parser_group = parser.add_mutually_exclusive_group(required=True)
    parser_group.add_argument("--json", help="JSON file path with list of samples to predict", type=str)
    parser_group.add_argument("--image", help="Image file path to predict", nargs="*", type=str,
                              action="append")
    parser_group.add_argument("--csv", help="CSV path without header to predict", type=str)


def run_command(args):
    if args.json is not None:
        samples = np.array(read_json(args.json))
    elif args.csv is not None:
        samples = pd.read_csv(args.csv, header=None)
    elif args.image is not None:
        # images can be a nested list so we flatten the list
        samples = np.array([np.array(imread_rgb(image)) for image in itertools.chain(*args.image)])
    else:
        raise ValueError("Unrecognized argument")

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
