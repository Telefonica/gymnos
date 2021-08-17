#
#
#   Atari env
#
#

import os
import gym
import fastdl

from supersuit import frame_stack_v1
from atari_py.import_roms import import_roms as _import_roms

from ...config import get_gymnos_home
from .wrappers import AtariWrapper, DiscreteActionsWrapper


def import_atari_roms():
    dir_path = os.path.join(get_gymnos_home(), "downloads", "atari")

    with fastdl.Parallel(prefer="threads") as p:

        hc_roms_download = p.download(
            url="http://obiwan.hi.inet/public/gymnos/atari/HC_ROMS.zip",
            dir_prefix=dir_path,
            extract=True
        )

        roms_download = p.download(
            url="http://obiwan.hi.inet/public/gymnos/atari/ROMS.zip",
            dir_prefix=os.path.join(get_gymnos_home(), "downloads", "atari"),
            extract=True
        )

        hc_roms_download.get()
        roms_download.get()

    _import_roms(dir_path)


def Atari(id, use_wrapper=True, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False,
          frame_stack=0, include_actions=None, clip_reward=True):
    import_atari_roms()

    env = gym.make(id)

    if use_wrapper:
        if include_actions is not None:
            env = DiscreteActionsWrapper(env, include_actions)

        env = AtariWrapper(env, noop_max, frame_skip, screen_size, terminal_on_life_loss, clip_reward)

        if frame_stack > 0:
            env = frame_stack_v1(env, frame_stack)

    return env
