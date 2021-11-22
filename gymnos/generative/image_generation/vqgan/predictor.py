#
#
#   Predictor
#
#

from omegaconf import DictConfig

from ....base import BasePredictor, MLFlowRun


class VqganPredictor(BasePredictor):
    """
    TODO: docstring for predictor
    """

    def load(self, config: DictConfig, run: MLFlowRun, artifacts_dir: str):
        pass   # OPTIONAL: load model from MLFlow artifacts directory

    def predict(self, *args, **kwargs):
        pass   # TODO: prediction code. Define parameters
