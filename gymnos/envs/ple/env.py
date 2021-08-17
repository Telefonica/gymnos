#
#
#   Ple env
#
#

import gym
import gym_ple  # noqa: DO NOT REMOVE IMPORT

from supersuit import frame_stack_v1, color_reduction_v0


def PLE(name, use_wrapper: bool = False, frame_stack: int = 0):
    env = gym.make(name + "-v0")
    if use_wrapper:
        if frame_stack > 0:
            env = frame_stack_v1(color_reduction_v0(env), frame_stack)
    return env
