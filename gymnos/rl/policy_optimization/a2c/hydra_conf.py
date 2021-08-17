#
#
#   A2C Hydra configuration
#
#

import enum

from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from ...common.sb3_mixins_hydra_conf import SB3TrainerHydraConf


class A2CPolicy(enum.Enum):
    MLP = "MlpPolicy"
    CNN = "CnnPolicy"


@dataclass
class A2CHydraConf(SB3TrainerHydraConf):

    policy_kwargs: Optional[Dict[str, Any]] = None
    policy: A2CPolicy = A2CPolicy.MLP
    learning_rate: float = 0.001
    n_steps: int = 5
    discount_rate: float = 0.99
    gae_lambda: float = 1.0
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    rms_prop_eps: float = 1e-05
    use_rms_prop: bool = True
    use_sde: bool = False
    sde_sample_freq: int = - 1
    normalize_advantage: bool = False
    num_envs: int = 1
    device: str = "auto"
    seed: int = 0

    _target_: str = field(init=False, repr=False, default="gymnos.rl.policy_optimization.a2c."
                                                          "trainer.A2CTrainer")
