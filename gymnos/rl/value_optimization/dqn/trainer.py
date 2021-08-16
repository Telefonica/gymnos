#
#
#   Trainer
#
#

import gym

from stable_baselines3 import DQN
from dataclasses import dataclass

from ....base import BaseRLTrainer
from .hydra_conf import DQNHydraConf
from ...common.sb3_mixins import SB3Trainer


@dataclass
class DQNTrainer(SB3Trainer, DQNHydraConf, BaseRLTrainer):
    """
    TODO: docstring for trainer
    """

    def prepare_env(self, env_id):
        self.env_id = env_id

    def create_env(self):
        return gym.make(self.env_id)

    def create_model(self, env):
        return DQN(self.policy.value, env, self.learning_rate, self.buffer_size, self.learning_starts,
                   self.batch_size, self.tau, self.discount_rate, self.train_freq, self.gradient_steps,
                   self.replay_buffer_class, self.replay_buffer_kwargs, self.optimize_memory_usage,
                   self.target_update_interval, self.exploration_fraction, self.exploration_initial_eps,
                   self.exploration_final_eps, self.max_grad_norm, device=self.device, seed=self.seed,
                   policy_kwargs=self.policy_kwargs)
