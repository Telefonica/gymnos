#
#
#   Trainer
#
#

from dataclasses import dataclass

from ....base import BaseTrainer
from .hydra_conf import VqganHydraConf


@dataclass
class VqganTrainer(VqganHydraConf, BaseTrainer):
    """
    TODO: docstring for trainer
    """

    def prepare_data(self, root):
        pass   # OPTIONAL: do anything with your data

    def train(self):
        pass   # TODO: training code

    def test(self):
        pass   # OPTIONAL: test code
