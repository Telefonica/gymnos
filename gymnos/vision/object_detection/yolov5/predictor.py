#
#
#   Predictor
#
#

from ....base import BasePredictor


class Yolov5Predictor(BasePredictor):
    """
    TODO: docstring for predictor
    """

    def load(self, artifacts_dir):
        pass   # OPTIONAL: load model from MLFlow artifacts directory

    def predict(self, *args, **kwargs):
        pass   # TODO: prediction code. Define parameters
