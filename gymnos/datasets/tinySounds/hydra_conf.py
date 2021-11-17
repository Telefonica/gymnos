#
#
#   Tiny Sounds Hydra conf
#
#

from dataclasses import dataclass, field


@dataclass
class TinySoundsHydraConf:


    _target_: str = field(init=False, repr=False, default="gymnos.datasets.tinySounds.dataset.TinySounds")
