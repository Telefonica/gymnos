#
#
#   Tiny Sounds Hydra conf
#
#

from dataclasses import dataclass, field


@dataclass
class TinySoundsHydraConf:

    # TODO: add custom parameters

    _target_: str = field(init=False, repr=False, default="gymnos.datasets.tinySounds.dataset.TinySounds")
