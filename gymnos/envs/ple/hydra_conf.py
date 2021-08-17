#
#
#   Ple Hydra conf
#
#

from dataclasses import dataclass, field


@dataclass
class PLEHydraConf:

    name: str
    use_wrapper: bool = True
    frame_stack: int = 0

    _target_: str = field(init=False, repr=False, default="gymnos.envs.ple.env.PLE")
