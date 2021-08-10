#
#
#   A2C Hydra configuration
#
#

import enum

from typing import Optional
from dataclasses import dataclass, field


class A2CPolicy(enum.Enum):

    CNN = "cnn"


@dataclass
class A2CHydraConf:

    discount_rate: float = 0.99
    update_frequency: int = 1
    entropy_coef: float = 0.001
    value_coef: float = 0.5
    device: Optional[str] = None
    num_envs: int = 1
    asynchronous: bool = True
    num_train_episodes: int = 100
    policy: A2CPolicy = A2CPolicy.CNN

    _target_: str = field(init=False, repr=False, default="gymnos.rl.policy_optimization.a2c."
                                                          "trainer.A2CTrainer")
