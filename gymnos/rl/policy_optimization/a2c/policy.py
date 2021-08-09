#
#
#   Network
#
#

import gym
import torch.nn as nn

from typing import Union


class CNNPolicy(nn.Module):

    def __init__(self, env: Union[gym.Env, gym.vector.VectorEnv]):
        super().__init__()

        if isinstance(env, gym.vector.VectorEnv):
            action_space = env.single_action_space
            observation_space = env.single_observation_space
        else:
            action_space = env.action_space
            observation_space = env.observation_space

        num_actions = action_space.n
        channels = observation_space.shape[0]

        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()

        self.critic_fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.actor_fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        value = self.critic_fc(x)
        action = self.actor_fc(x)
        return value, action
