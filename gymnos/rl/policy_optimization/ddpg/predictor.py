#
#
#   Predictor
#
#

from stable_baselines3 import DDPG

from ...common.sb3_mixins import SB3Predictor
from ....base import BasePredictor, MLFlowRun


class DDPGPredictor(SB3Predictor, BasePredictor):
    """
    TODO: docstring for predictor
    """

    def __init__(self, device: str = "auto"):
        self.device = device

    def load_model(self, path):
        return DDPG.load(path, device=self.device)
