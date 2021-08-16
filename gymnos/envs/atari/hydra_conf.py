#
#
#   Atari Hydra conf
#
#

from typing import Optional, List
from dataclasses import dataclass, field


@dataclass
class AtariHydraConf:

    id: str
    use_wrapper: bool = True
    noop_max: int = 30
    frame_skip: int = 4
    screen_size: int = 84
    clip_reward: bool = True
    terminal_on_life_loss: bool = False
    frame_stack: int = 0
    include_actions: Optional[List[int]] = None

    _target_: str = field(init=False, repr=False, default="gymnos.envs.atari.env.Atari")
