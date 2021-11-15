#
#
#   Predictor
#
#

from omegaconf import DictConfig

from ....base import BasePredictor, MLFlowRun
import os
import tensorflow as tf
import joblib
import numpy as np

class NeuralNetworkClassifierPredictor(BasePredictor):
    """
    TODO: docstring for predictor
    """

    def load(self, config: DictConfig, run: MLFlowRun, artifacts_dir: str):
        # Load model
        model_path = os.path.join(artifacts_dir,'nn_classifier_cal_intake.h5')
        self.nn_classifier = tf.keras.models.load_model(model_path)
        # Load scaler
        scaler_path = os.path.join(artifacts_dir,'scaler')
        self.scaler = joblib.load(scaler_path)

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
        # Normalize input using the loaded scaler
        X = self.scaler.transform(X)
        predictions = self.nn_classifier.predict(X)
        self.pred = []
        for i in range(len(predictions)):
            self.pred.append(np.argmax(predictions[i])) # Gives the 0,1 or 2
