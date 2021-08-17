#
#
#   Random Hydra configuration
#
#

from dataclasses import dataclass, field


@dataclass
class RandomHydraConf:

    num_train_timesteps: int = 1_000
    num_test_episodes: int = 1
    log_interval: int = 1_000

    _target_: str = field(init=False, repr=False, default="gymnos.rl.misc.random."
                                                          "trainer.RandomTrainer")
