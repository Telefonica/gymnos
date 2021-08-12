#
#
#   Trainer
#
#

import sys

from dataclasses import dataclass
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import Logger, HumanOutputFormat

from ....base import BaseRLTrainer
from .hydra_conf import A2CHydraConf
from ...common.sb3_mlflow import MlflowKVWriter


@dataclass
class A2CTrainer(A2CHydraConf, BaseRLTrainer):
    """
    TODO: docstring for trainer
    """

    def __post_init__(self):
        ...

    def prepare_env(self, env_id):
        self._env_id = env_id

    def train(self):
        env = make_vec_env(self._env_id, n_envs=self.num_envs)

        output_formats = [
            MlflowKVWriter()
        ]

        if self.verbose:
            output_formats.insert(0, HumanOutputFormat(sys.stdout))

        mlflow_logger = Logger(folder=None, output_formats=output_formats)

        model = A2C(self.policy.value, env)
        model.set_logger(mlflow_logger)
        model.learn(total_timesteps=self.train_timesteps)

    def test(self):
        pass   # OPTIONAL: test code
