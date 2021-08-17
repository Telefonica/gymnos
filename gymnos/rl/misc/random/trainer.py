#
#
#   Trainer
#
#

import gym
import mlflow
import pickle
import numpy as np

from tqdm import trange
from dataclasses import dataclass
from collections import defaultdict

from ....base import BaseRLTrainer
from .hydra_conf import RandomHydraConf


@dataclass
class RandomTrainer(RandomHydraConf, BaseRLTrainer):
    """
    TODO: docstring for trainer
    """

    def prepare_env(self, env_id):
        self.env_id = env_id

    def train(self):
        env = gym.make(self.env_id)

        episode = 0
        timestep = 0
        ep_rewards = []
        metrics = defaultdict(list)

        _ = env.reset()

        while timestep < self.num_train_timesteps:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)

            timestep += 1

            ep_rewards.append(reward)

            if (timestep % self.log_interval) == 0:
                log_metrics = {
                    "train/total_timesteps": timestep,
                    "train/total_episodes": episode,
                    "train/ep_reward": np.mean(metrics["ep_reward"]),
                    "train/ep_length": np.mean(metrics["ep_length"])
                }

                for metric_name, metric_value in log_metrics.items():
                    print(f"{metric_name}: {metric_value}")

                print("*" * 20)

                mlflow.log_metrics(log_metrics, timestep)

                metrics.clear()

            if done:
                metrics["ep_reward"].append(np.sum(ep_rewards))
                metrics["ep_length"].append(len(ep_rewards))

                episode += 1
                ep_rewards = []

                _ = env.reset()

        with open("action_space.pkl", "wb") as fp:
            pickle.dump(env.action_space, fp)

        mlflow.log_artifact("action_space.pkl")

    def test(self):
        env = gym.make(self.env_id)

        global_step = 0

        for _ in trange(self.num_test_episodes):
            ep_reward = 0.0
            ep_length = 0

            done = False
            _ = env.reset()

            while not done:
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)

                ep_length += 1
                ep_reward += reward

            global_step += ep_length

            mlflow.log_metrics({
                "test/ep_reward": ep_reward,
                "test/ep_length": ep_length
            }, global_step)
