#
#
#   Predictor
#
#

from ....base import BasePredictor
from ...common.sb3_mixins import SB3Predictor

from stable_baselines3 import A2C


class A2CPredictor(SB3Predictor, BasePredictor):
    """
    TODO: docstring for predictor
    """

    def __init__(self, device: str = "auto"):
        self.device = device

    def load_model(self, path):
        return A2C.load(path, device=self.device)
