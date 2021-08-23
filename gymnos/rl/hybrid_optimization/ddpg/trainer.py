#
#
#   Trainer
#
#

import gym

from dataclasses import dataclass
from stable_baselines3 import DDPG

from ....base import BaseRLTrainer
from .hydra_conf import DDPGHydraConf
from ...common.sb3_mixins import SB3Trainer


@dataclass
class DDPGTrainer(SB3Trainer, DDPGHydraConf, BaseRLTrainer):
    """
    TODO: docstring for trainer
    """

    def prepare_env(self, env_id):
        self.env_id = env_id

    def create_env(self):
        return gym.make(self.env_id)

    def create_model(self, env):
        return DDPG(self.policy.value, env, self.learning_rate, self.buffer_size, self.learning_starts,
                    self.batch_size, self.tau, self.discount_rate, self.train_freq, self.gradient_steps,
                    self.action_noise, self.replay_buffer_class, self.replay_buffer_kwargs,
                    self.optimize_memory_usage, device=self.device, seed=self.seed, policy_kwargs=self.policy_kwargs)
