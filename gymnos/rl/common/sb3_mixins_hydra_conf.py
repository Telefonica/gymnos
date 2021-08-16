#
#
#   Stable-baselines 3 Hydra conf
#
#

import enum

from typing import Optional
from dataclasses import dataclass


class SaveStrategy(enum.Enum):
    BEST = enum.auto()
    LAST = enum.auto()


@dataclass()
class SB3TrainerHydraConf:

    num_train_timesteps: int
    verbose: bool = True
    save_strategy: SaveStrategy = SaveStrategy.LAST
    eval_freq: int = 10_000
    n_eval_episodes: int = 5
    max_num_train_episodes: Optional[int] = None
    stop_training_reward_threshold: Optional[float] = None
    log_interval: int = 100
    num_test_episodes: int = 1
