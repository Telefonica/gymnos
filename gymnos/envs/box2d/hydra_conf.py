#
#
#   Box2d Hydra conf
#
#

from dataclasses import dataclass, field


@dataclass
class Box2dHydraConf:

    id: str

    _target_: str = field(init=False, repr=False, default="gymnos.envs.box2d.env.Box2d")
