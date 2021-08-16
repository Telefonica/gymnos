#
#
#   Box2d env
#
#

import gym

from dataclasses import dataclass

from .hydra_conf import Box2dHydraConf


def Box2d(id):
    env = gym.make(id)
    return env
