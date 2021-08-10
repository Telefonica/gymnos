#
#
#   Trainer
#
#

import gym
import mlflow
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

from tqdm import tqdm
from dataclasses import dataclass

from .utils import discount
from .policy import CNNPolicy
from .wrapper import cnn_wrapper
from ....base import BaseRLTrainer
from .hydra_conf import A2CHydraConf, A2CPolicy


@dataclass
class A2CTrainer(A2CHydraConf, BaseRLTrainer):
    """
    TODO: docstring for trainer
    """

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def prepare_env(self, env_id):
        self._env_id = env_id

    def train(self):
        wrappers = None
        if self.policy == A2CPolicy.CNN:
            wrappers = cnn_wrapper

        env = gym.vector.make(self._env_id, self.num_envs, self.asynchronous, wrappers)

        if self.policy == A2CPolicy.CNN:
            policy_network = CNNPolicy(env)
        else:
            raise ValueError(f"Unexpected policy {self.policy}")

        policy_network.to(self.device)

        policy_optimizer = optim.Adam(policy_network.parameters())

        obs = env.reset()

        global_step = 0
        current_episode = 0

        pbar = tqdm(total=self.num_train_episodes)

        def t(x):
            return torch.from_numpy(x).to(self.device)

        episode_rewards = np.zeros(env.num_envs)
        episode_steps = np.zeros(env.num_envs)

        def init_n_step():
            return dict(
                step=0,
                entropy=0.0,
                rewards=torch.empty([self.update_frequency, env.num_envs], dtype=torch.float32, device=self.device),
                dones=torch.empty([self.update_frequency, env.num_envs], dtype=torch.int32, device=self.device),
                state_values=torch.empty([self.update_frequency, env.num_envs], dtype=torch.float32, device=self.device),
                log_probs=torch.empty([self.update_frequency, env.num_envs], dtype=torch.float32, device=self.device)
            )

        n_step = init_n_step()

        while current_episode < self.num_train_episodes:
            state_value, action_logits = policy_network(t(obs))
            action_proba = F.softmax(action_logits, dim=-1)

            action_dist = distributions.Categorical(action_proba)
            action_pred = action_dist.sample()

            action = action_pred.detach().cpu().numpy()
            next_obs, reward, done, info = env.step(action)

            n_step["rewards"][n_step["step"]] = t(reward)
            n_step["dones"][n_step["step"]] = t(done)
            n_step["state_values"][n_step["step"]] = state_value.flatten()
            n_step["entropy"] += action_dist.entropy().mean()
            n_step["log_probs"][n_step["step"]] = action_dist.log_prob(action_pred)

            if n_step["step"] == (self.update_frequency - 1):
                v_next_s, _ = policy_network(t(next_obs))
                v_next_s = v_next_s.view(1, -1)

                with torch.no_grad():
                    returns = discount(n_step["rewards"], n_step["dones"], self.discount_rate, v_next_s).squeeze()

                advantage = returns - n_step["state_values"]

                critic_loss = advantage ** 2
                actor_loss = -n_step["log_probs"] * advantage.detach()

                # Mean for step and worker
                critic_loss = critic_loss.mean(0).mean()
                actor_loss = actor_loss.mean(0).mean()

                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * n_step["entropy"]

                mlflow.log_metrics({
                    "train/loss": loss.item(),
                    "train/actor_loss": actor_loss.item(),
                    "train/critic_loss": critic_loss.item()
                }, global_step)

                policy_optimizer.zero_grad()

                loss.backward()

                policy_optimizer.step()

                n_step = init_n_step()
            else:
                n_step["step"] += 1

            obs = next_obs

            episode_rewards += reward
            episode_steps += 1

            done_envs = np.nonzero(done)[0]

            for i, done_env in enumerate(done_envs):
                mlflow.log_metrics({
                    "episode_reward": episode_rewards[done_env],
                    "episode_steps": episode_steps[done_env]
                }, global_step + i)

            global_step += env.num_envs

            episode_steps[done_envs] = 0
            episode_rewards[done_envs] = 0.0

            current_episode += len(done_envs)

            pbar.update(len(done_envs))

            if len(done_envs) > 0:
                mlflow.log_metric("train/episode", current_episode, global_step)

        torch.save(policy_network.state_dict(), "policy_state_dict.pth")
        mlflow.log_artifact("policy_state_dict.pth")

    def test(self):
        pass   # OPTIONAL: test code
