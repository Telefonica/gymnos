#
#
#   Trainer
#
#

from dataclasses import dataclass
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from ....base import BaseRLTrainer
from .hydra_conf import PPOHydraConf
from ...common.sb3_mixins import SB3Trainer


@dataclass
class PPOTrainer(SB3Trainer, PPOHydraConf, BaseRLTrainer):
    """
    TODO: docstring for trainer
    """

    def prepare_env(self, env_id):
        self.env_id = env_id

    def create_env(self):
        return make_vec_env(self.env_id, self.num_envs, self.seed)

    def create_model(self, env):
        return PPO(self.policy.value, env, self.learning_rate, self.n_steps, self.batch_size,
                   self.n_epochs, self.discount_rate, self.gae_lambda, self.clip_range, self.clip_range_value,
                   self.entropy_coef, self.value_coef, self.max_grad_norm, self.use_sde, self.sde_sample_freq,
                   self.target_kl, device=self.device, seed=self.seed, policy_kwargs=self.policy_kwargs)
