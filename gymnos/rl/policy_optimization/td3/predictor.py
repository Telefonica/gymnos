#
#
#   Predictor
#
#

from stable_baselines3 import TD3

from ....base import BasePredictor
from ...common.sb3_mixins import SB3Predictor


class TD3Predictor(SB3Predictor, BasePredictor):
    """
    TODO: docstring for predictor
    """

    def __init__(self, device: str = "auto"):
        self.device = device

    def load_model(self, path):
        return TD3.load(path, device=self.device)
