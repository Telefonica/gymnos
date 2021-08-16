#
#
#   Predictor
#
#

from ....base import BasePredictor
from ...common.sb3_mixins import SB3Predictor

from stable_baselines3 import DQN


class DQNPredictor(SB3Predictor, BasePredictor):
    """
    TODO: docstring for predictor
    """

    def __init__(self, device: str = "auto"):
        self.device = device

    def load_model(self, path):
        return DQN.load(path, device=self.device)
