#
#
#   Atari gymnos conf
#
#

from .hydra_conf import AtariHydraConf

hydra_conf = AtariHydraConf

reward_threshold = None  # The reward threshold before the task is considered solved
nondeterministic = False  # Whether this environment is non-deterministic even after seeding
max_episode_steps = None  # The maximum number of steps that an episode can consist of

pip_dependencies = [
    "gym[atari]",
    "atari-py",
    "supersuit"
]

apt_dependencies = [
    "libgl1-mesa-glx"
]
