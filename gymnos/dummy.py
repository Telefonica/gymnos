#
#
#   Dummy
#
#

from dataclasses import dataclass, field

from .base import BaseDataset


@dataclass
class DummyDatasetHydraConf:

    _target_: str = field(init=False, default="gymnos.dummy.DummyDataset")


class DummyDataset(BaseDataset):

    def __call__(self, root):
        pass


@dataclass
class DummyTrainerHydraConf:

    _target_: str = field(init=False, default="gymnos.dummy.DummyTrainer")


class DummyTrainer:

    def setup(self, root):
        pass

    def train(self):
        pass

    def test(self):
        pass
