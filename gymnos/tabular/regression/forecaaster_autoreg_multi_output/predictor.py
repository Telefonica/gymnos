#
#
#   Predictor
#
#

from omegaconf import DictConfig

from ....base import BasePredictor, MLFlowRun
import os
import joblib


class ForecaasterAutoregMultiOutputPredictor(BasePredictor):
    """
    TODO: docstring for predictor
    """

    def load(self, config: DictConfig, run: MLFlowRun, artifacts_dir: str):
        model_path = os.path.join(artifacts_dir,'forecaster_r')
        self.rf_model = joblib.load(model_path)

    def predict(self, X):
        self.pred = self.forecaster_r.predict(X)
