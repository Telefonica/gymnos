#
#
#   Dummy model
#
#

from dataclasses import field, dataclass


@dataclass
class DummyHydraConf:

    _target_: str = field(init=False, default="gymnos.dummy.__model__.DummyTrainer")


hydra_conf = DummyHydraConf

pip_dependencies = []

apt_dependencies = []


class DummyTrainer:

    def setup(self, root):
        pass

    def train(self):
        pass

    def test(self):
        pass
