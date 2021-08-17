#
#
#   Predictor
#
#

from stable_baselines3 import SAC

from ....base import BasePredictor
from ...common.sb3_mixins import SB3Predictor


class SACPredictor(SB3Predictor, BasePredictor):
    """
    TODO: docstring for predictor
    """

    def __init__(self, device: str = "auto"):
        self.device = device

    def load_model(self, path):
        return SAC.load(path, device=self.device)
