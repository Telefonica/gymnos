#
#
#   Ple env
#
#

import gym
import gym_ple  # noqa: DO NOT REMOVE IMPORT


def PLE(name):
    return gym.make(name + "-v0")
