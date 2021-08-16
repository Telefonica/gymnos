#
#
#   Trainer
#
#

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage

from ....base import BaseRLTrainer
from .hydra_conf import A2CHydraConf, A2CPolicy
from ...common.sb3_mixins import SB3Trainer

from dataclasses import dataclass


@dataclass
class A2CTrainer(SB3Trainer, A2CHydraConf, BaseRLTrainer):
    """
    TODO: docstring for trainer
    """

    def prepare_env(self, env_id):
        self.env_id = env_id

    def create_env(self):
        venv = make_vec_env(self.env_id, n_envs=self.num_envs, seed=self.seed)
        if self.policy == A2CPolicy.CNN:
            venv = VecTransposeImage(venv)
        return venv

    def create_model(self, env):
        return A2C(self.policy.value, env, seed=self.seed, device=self.device, learning_rate=self.learning_rate,
                   n_steps=self.n_steps, gamma=self.discount_rate, gae_lambda=self.gae_lambda,
                   ent_coef=self.entropy_coef, vf_coef=self.value_coef, max_grad_norm=self.max_grad_norm,
                   rms_prop_eps=self.rms_prop_eps, use_rms_prop=self.use_rms_prop, use_sde=self.use_sde,
                   sde_sample_freq=self.sde_sample_freq, normalize_advantage=self.normalize_advantage,
                   policy_kwargs=self.policy_kwargs)
