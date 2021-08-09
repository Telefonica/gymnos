#
#
#  Dummy env
#
#

import gym

from dataclasses import dataclass, field


@dataclass
class DummyEnvHydraConf:

    _target_: str = field(init=False, repr=False, default="gymnos.dummy.__env__.DummyEnv")


@dataclass
class DummyEnv(DummyEnvHydraConf, gym.Env):

    metadata = {'render.modes': ['human']}

    @property
    def action_space(self) -> gym.Space:
        return gym.Space()

    @property
    def observation_space(self) -> gym.Space:
        return gym.Space()

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human', close=False):
        pass


hydra_conf = DummyEnvHydraConf

reward_threshold = None  # The reward threshold before the task is considered solved
nondeterministic = False  # Whether this environment is non-deterministic even after seeding
max_episode_steps = None  # The maximum number of steps that an episode can consist of

pip_dependencies = []

apt_dependencies = []
