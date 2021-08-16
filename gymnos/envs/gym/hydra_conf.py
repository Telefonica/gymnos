#
#
#   Gym Hydra conf
#
#

from dataclasses import dataclass, field


@dataclass
class GymHydraConf:

    id: str

    _target_: str = field(init=False, repr=False, default="gymnos.envs.gym.env.Gym")
