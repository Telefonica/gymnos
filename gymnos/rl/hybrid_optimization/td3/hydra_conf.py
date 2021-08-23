#
#
#   Td3 Hydra configuration
#
#
import enum

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from ...common.sb3_mixins_hydra_conf import SB3TrainerHydraConf


class TD3Policy(enum.Enum):
    MLP = "MlpPolicy"
    CNN = "CnnPolicy"
    MULTI_INPUT = "MultiInputPolicy"


@dataclass
class TD3HydraConf(SB3TrainerHydraConf):

    policy: TD3Policy = TD3Policy.MLP
    policy_kwargs: Optional[Dict[str, Any]] = None
    learning_rate: float = 0.001
    buffer_size: int = 1_000_000
    learning_starts: int = 100
    batch_size: int = 100
    tau: float = 0.005
    discount_rate: float = 0.99
    train_freq: List = (1, 'episode')
    gradient_steps: int = - 1
    action_noise = None  # FIXME
    replay_buffer_class = None  # FIXME
    replay_buffer_kwargs = None  # FIXME
    optimize_memory_usage: bool = False
    policy_delay: int = 2
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5
    device: str = "auto"
    seed: int = 0

    _target_: str = field(init=False, repr=False, default="gymnos.rl.policy_optimization.td3."
                                                          "trainer.TD3Trainer")
