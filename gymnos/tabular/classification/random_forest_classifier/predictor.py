#
#
#   Predictor
#
#

from omegaconf import DictConfig

from ....base import BasePredictor, MLFlowRun
import os
import joblib

class RandomForestClassifierPredictor(BasePredictor):
    """
    TODO: docstring for predictor
    """

    def load(self, config: DictConfig, run: MLFlowRun, artifacts_dir: str):
        # Load model
        model_path = os.path.join(artifacts_dir,'rf_model')
        self.rf_model = joblib.load(model_path)

    def predict(self, X):
        '''
        Input:
            X: Array containing the Age, Gender and Activity as they are defined in cal_intake.csv.
               X can also be a list for making multiple predictions
        Output:
            pred: Will be the alimentation classification by calories. Categories are the same defined in cal_intake.csv
                  (0 if dayly calories intaked have been under the adequate values for a certain age, gender and activity,
                   1 if the dayly calories intaked have been the adequate,
                   2 if the dayly calories intaked have been over the adequate).
                   pred will be a single value or a list depending the components of the input X
        '''
        self.pred = self.rf_model.predict(X)