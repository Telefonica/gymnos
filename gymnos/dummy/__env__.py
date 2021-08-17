#
#
#  Dummy env
#
#


from dataclasses import dataclass, field


@dataclass
class DummyEnvHydraConf:

    _target_: str = field(init=False, repr=False, default="gymnos.dummy.__env__.DummyEnv")


def DummyEnv():
    import gym
    return gym.Env()


hydra_conf = DummyEnvHydraConf

reward_threshold = None  # The reward threshold before the task is considered solved
nondeterministic = False  # Whether this environment is non-deterministic even after seeding
max_episode_steps = None  # The maximum number of steps that an episode can consist of

pip_dependencies = []

apt_dependencies = []
