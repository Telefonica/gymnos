#
#
#   Ple Hydra conf
#
#

from dataclasses import dataclass, field


@dataclass
class PLEHydraConf:

    name: str

    _target_: str = field(init=False, repr=False, default="gymnos.envs.ple.env.PLE")
