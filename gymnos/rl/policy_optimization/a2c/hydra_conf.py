#
#
#   A2C Hydra configuration
#
#

import enum

from typing import Optional
from dataclasses import dataclass, field


class A2CPolicy(enum.Enum):
    MLP = "MlpPolicy"
    CNN = "CnnPolicy"


@dataclass
class A2CHydraConf:

    train_timesteps: int
    policy: A2CPolicy = A2CPolicy.MLP
    num_envs: int = 1
    verbose: bool = True

    _target_: str = field(init=False, repr=False, default="gymnos.rl.policy_optimization.a2c."
                                                          "trainer.A2CTrainer")
