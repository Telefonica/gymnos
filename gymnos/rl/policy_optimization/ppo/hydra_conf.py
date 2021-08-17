#
#
#   Ppo Hydra configuration
#
#
import enum

from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from ...common.sb3_mixins_hydra_conf import SB3TrainerHydraConf


class PPOPolicy(enum.Enum):
    MLP = "MlpPolicy"
    CNN = "CnnPolicy"
    MULTI_INPUT = "MultiInputPolicy"


@dataclass
class PPOHydraConf(SB3TrainerHydraConf):

    policy: PPOPolicy = PPOPolicy.MLP
    num_envs: int = 1
    seed: int = 0
    policy_kwargs: Optional[Dict[str, Any]] = None
    learning_rate: float = 0.0003
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    discount_rate: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_value: Optional[float] = None
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_sde: bool = False
    sde_sample_freq: int = - 1
    target_kl: Optional[float] = None
    device: str = "auto"

    _target_: str = field(init=False, repr=False, default="gymnos.rl.policy_optimization.ppo."
                                                          "trainer.PPOTrainer")
