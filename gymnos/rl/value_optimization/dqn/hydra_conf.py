#
#
#   Dqn Hydra configuration
#
#

import enum

from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from ...common.sb3_mixins_hydra_conf import SB3TrainerHydraConf


class DQNPolicy(enum.Enum):
    MLP = "MlpPolicy"
    CNN = "CnnPolicy"
    MULTI_INPUT = "MultiInputPolicy"


@dataclass
class DQNHydraConf(SB3TrainerHydraConf):

    policy: DQNPolicy = DQNPolicy.MLP
    policy_kwargs: Optional[Dict[str, Any]] = None
    learning_rate: float = 0.0001
    buffer_size: int = 1_000_000
    learning_starts: int = 50_000
    batch_size: int = 32
    tau: float = 1.0
    discount_rate: float = 0.99
    train_freq: int = 4
    gradient_steps: int = 1
    replay_buffer_class = None  # FIXME
    replay_buffer_kwargs = None  # FIXME
    optimize_memory_usage: bool = False
    target_update_interval: int = 10_000
    exploration_fraction: float = 0.1
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    max_grad_norm: int = 10
    seed: int = 0
    device: str = "auto"

    _target_: str = field(init=False, repr=False, default="gymnos.rl.value_optimization.dqn."
                                                          "trainer.DQNTrainer")
