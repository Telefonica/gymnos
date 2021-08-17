#
#
#   Atari env
#
#

import gym

from supersuit import frame_stack_v1

from .utils import import_atari_roms
from .wrappers import AtariWrapper, DiscreteActionsWrapper


import_atari_roms()


def Atari(id, use_wrapper=True, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False,
          frame_stack=0, include_actions=None, clip_reward=True):
    env = gym.make(id)

    if use_wrapper:
        if include_actions is not None:
            env = DiscreteActionsWrapper(env, include_actions)

        env = AtariWrapper(env, noop_max, frame_skip, screen_size, terminal_on_life_loss, clip_reward)

        if frame_stack > 0:
            env = frame_stack_v1(env, frame_stack)

    return env
