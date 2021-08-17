#
#
#   Trainer
#
#

import gym

from dataclasses import dataclass
from stable_baselines3 import TD3

from ....base import BaseRLTrainer
from .hydra_conf import TD3HydraConf


@dataclass
class TD3Trainer(TD3HydraConf, BaseRLTrainer):
    """
    TODO: docstring for trainer
    """

    def prepare_env(self, env_id):
        self.env_id = env_id

    def create_env(self):
        return gym.make(self.env_id)

    def create_model(self, env):
        return TD3(self.policy.value, env, self.learning_rate, self.buffer_size, self.learning_starts,
                   self.batch_size, self.tau, self.discount_rate, self.train_freq, self.gradient_steps,
                   self.action_noise, self.replay_buffer_class, self.replay_buffer_kwargs,
                   self.optimize_memory_usage, self.policy_delay, self.target_policy_noise,
                   self.target_noise_clip, device=self.device, seed=self.seed, policy_kwargs=self.policy_kwargs)
