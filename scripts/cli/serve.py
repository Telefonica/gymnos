#
#
#   Serve CLI app
#
#

import json
import numpy as np

from gymnos.trainer import Trainer
from gymnos.datasets.dataset import ClassLabel
from gymnos.utils.lazy_imports import lazy_imports as lazy


def add_arguments(parser):
    parser.add_argument("saved_trainer", help="Saved trainer file", type=str)
    parser.add_argument("--host", help="Hostname to listen to.", type=str, required=False)
    parser.add_argument("--port", help="Port to listen to.", type=int, required=False)
    parser.add_argument("--debug", help="Enable or disable debug mode", default=False, required=False,
                        action="store_true")


def run_command(args):
    class FlaskNumpyEncoder(lazy.flask.json.JSONEncoder):
        """
        Flask Json Encoder to handle Numpy arrays
        """

        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    app = lazy.flask.Flask(__name__)
    app.json_encoder = FlaskNumpyEncoder

    @app.route("/", methods=["GET"])
    def info():
        # FIXME: the trainer should be loaded before every request but there are issues combining tf with flask requests
        #           because tf session is not on the same thread as the request
        trainer = Trainer.load(args.saved_trainer)
        return lazy.flask.jsonify(trainer.to_dict())

    @app.route("/", methods=["POST"])
    def predict():
        if not lazy.flask.request.is_json:
            return lazy.flask.jsonify(error="Request body must be JSON"), 400

        trainer = Trainer.load(args.saved_trainer)

        try:
            response = dict(predictions=trainer.predict(lazy.flask.request.get_json()))
        except Exception as e:
            return lazy.flask.jsonify(error="Prediction failed: {}".format(e)), 400

        try:
            response["probabilities"] = trainer.predict_proba(lazy.flask.request.json)
        except NotImplementedError:
            pass
        except Exception as e:
            return lazy.flask.jsonify(error="Prediction for probabilities failed: {}".format(e)), 400

        labels = trainer.dataset.dataset.labels_info
        if isinstance(labels, ClassLabel):
            response["classes"] = dict(
                names=labels.names,
                total=labels.num_classes
            )

        return lazy.flask.jsonify(response)

    app.run(host=args.host, port=args.port, debug=args.debug)
