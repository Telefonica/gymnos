#
#
#   Sac Hydra configuration
#
#
import enum

from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from ...common.sb3_mixins_hydra_conf import SB3TrainerHydraConf


class SACPolicy(enum.Enum):
    MLP = "MlpPolicy"
    CNN = "CnnPolicy"
    MULTI_INPUT = "MultiInputPolicy"


@dataclass
class SACHydraConf(SB3TrainerHydraConf):

    policy: SACPolicy = SACPolicy.MLP
    policy_kwargs: Optional[Dict[str, Any]] = None
    learning_rate: float = 0.0003
    buffer_size: int = 1000000
    learning_starts: int = 100
    batch_size: int = 256
    tau: float = 0.005
    discount_rate: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    action_noise = None  # FIXME
    replay_buffer_class = None  # FIXME
    replay_buffer_kwargs = None  # FIXME
    optimize_memory_usage: bool = False
    entropy_coef: str = 'auto'  # FIXME
    target_update_interval: int = 1
    target_entropy = 'auto'  # FIXME
    use_sde: bool = False
    sde_sample_freq: int = -1
    use_sde_at_warmup: bool = False
    seed: int = 0
    device: str = "auto"

    _target_: str = field(init=False, repr=False, default="gymnos.rl.hybrid_optimization.sac."
                                                          "trainer.SACTrainer")
