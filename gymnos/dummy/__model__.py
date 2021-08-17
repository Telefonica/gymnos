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

    def prepare_env(self, env_id):
        import gym
        gym.make(env_id)

    def prepare_data(self, root):
        pass

    def train(self):
        pass

    def test(self):
        pass
