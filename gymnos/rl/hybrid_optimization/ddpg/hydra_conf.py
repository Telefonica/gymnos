#
#
#   Ddpg Hydra configuration
#
#
import enum

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from ...common.sb3_mixins_hydra_conf import SB3TrainerHydraConf


class DDPGPolicy(enum.Enum):
    MLP = "MlpPolicy"
    CNN = "CnnPolicy"
    MULTI_INPUT = "MultiInputPolicy"


@dataclass
class DDPGHydraConf(SB3TrainerHydraConf):

    policy: DDPGPolicy = DDPGPolicy.MLP
    num_envs: int = 1
    seed: int = 0
    policy_kwargs: Optional[Dict[str, Any]] = None
    learning_rate: float = 0.001
    buffer_size: int = 1000000
    learning_starts: int = 100
    batch_size: int = 100
    tau: float = 0.005
    discount_rate: float = 0.99
    train_freq: List = (1, 'episode')
    gradient_steps: int = -1
    action_noise = None   # FIXME
    replay_buffer_class = None  # FIXME
    replay_buffer_kwargs = None  # FIXME
    optimize_memory_usage: bool = False
    device: str = "auto"

    _target_: str = field(init=False, repr=False, default="gymnos.rl.policy_optimization.ddpg."
                                                          "trainer.DDPGTrainer")
