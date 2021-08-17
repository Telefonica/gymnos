#
#
#   Ple gymnos conf
#
#

from .hydra_conf import PLEHydraConf

hydra_conf = PLEHydraConf

reward_threshold = None  # The reward threshold before the task is considered solved
nondeterministic = False  # Whether this environment is non-deterministic even after seeding
max_episode_steps = None  # The maximum number of steps that an episode can consist of

pip_dependencies = [
    "ple @ git+https://github.com/ntasfi/PyGame-Learning-Environment.git",
    "gym_ple @ git+https://github.com/lusob/gym-ple.git#egg=gym_ple",
    "pygame"
]

apt_dependencies = [
]
