#
#
#   Predict
#
#

import os
import base64
import argparse
import numpy as np
from pprint import pprint
from PIL import Image
from io import BytesIO
from lib.core.model import Model
from lib.core.dataset import Dataset
from lib.utils.io_utils import read_from_json
from bin.scripts.gymnosd import read_preferences

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prediction_config", help="Prediction config JSON file", required=True)
    parser.add_argument("-e", "--execution_dir", help="Execution directory to restore model and pipeline",
                        required=True)

    args = parser.parse_args()

    config = read_preferences()


    prediction_config = read_from_json(args.prediction_config, with_comments_support=False)

    model_dir = os.path.join(args.execution_dir, config["trained_model_dir"])
    preprocessors_filepath = os.path.join(args.execution_dir, config["trained_preprocessors_filename"])
    training_config_filepath = os.path.join(args.execution_dir, config["training_config_filename"])

    training_config = read_from_json(training_config_filepath, with_comments_support=True)

    model = Model(**training_config["model"])
    dataset = Dataset(**training_config["dataset"])

    model.model.restore(model_dir)
    dataset.preprocessors.restore(preprocessors_filepath)

    samples = []
    for sample in prediction_config["samples"]:
        if isinstance(sample, dict):
            if "image_bytes" in sample:
                sample = Image.open(BytesIO(base64.b64decode(sample["image_bytes"]["b64"]))).convert('RGB')
                sample = np.asarray(sample)
            else:
                raise ValueError("Supported keys: {}".format("image_bytes"))

        samples.append(sample)

    samples = np.array(samples)

    samples = dataset.preprocessors.transform(samples)
    predictions = model.model.predict(samples)

    pprint({
        "predictions": predictions
    })
