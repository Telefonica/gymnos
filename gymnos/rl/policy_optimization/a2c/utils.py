#
#
#   Utils
#
#

import numpy as np

from typing import Sequence

import torch


def range_like(x):
    return range(len(x))


def discount(rewards: Sequence, dones: Sequence, discount_rate: float = 0.9, next_value: float = 0.0):
    R = next_value
    discounted_rewards = []
    for step in reversed(range_like(rewards)):
        R = rewards[step] + discount_rate * R * (1 - dones[step])
        discounted_rewards.insert(0, R)

    if isinstance(rewards, np.ndarray):
        return np.array(discounted_rewards)
    elif torch.is_tensor(rewards):
        return torch.stack(discounted_rewards)
    else:
        return discounted_rewards
