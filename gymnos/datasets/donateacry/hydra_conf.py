#
#
#   Donateacry Hydra conf
#
#

from dataclasses import dataclass, field


@dataclass
class DonateacryHydraConf:

    # TODO: add custom parameters

    _target_: str = field(init=False, repr=False, default="gymnos.datasets.donateacry.dataset.Donateacry")
