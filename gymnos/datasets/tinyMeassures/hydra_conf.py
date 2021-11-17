#
#
#   Tiny Meassures Hydra conf
#
#

from dataclasses import dataclass, field


@dataclass
class TinyMeassuresHydraConf:


    _target_: str = field(init=False, repr=False,
                          default="gymnos.datasets.tinyMeassures.dataset.TinyMeassures")
