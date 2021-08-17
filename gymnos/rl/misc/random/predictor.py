#
#
#   Predictor
#
#

import os
import pickle

from ....base import BasePredictor


class RandomPredictor(BasePredictor):
    """
    TODO: docstring for predictor
    """

    def load(self, config, run, artifacts_dir):
        with open(os.path.join(artifacts_dir, "action_space.pkl"), "rb") as fp:
            self._action_space = pickle.load(fp)

    def predict(self, obs):
        return self._action_space.sample()
