#
#
#   Paintings Hydra conf
#
#

from dataclasses import dataclass, field


@dataclass
class PaintingsHydraConf:

    # TODO: add custom parameters
    force_extract: bool = False

    _target_: str = field(init=False, repr=False, default="gymnos.datasets.paintings.dataset.Paintings")
