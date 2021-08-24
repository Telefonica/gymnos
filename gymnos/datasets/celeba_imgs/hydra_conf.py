#
#
#   Celeba Imgs Hydra conf
#
#

from dataclasses import dataclass, field


@dataclass
class CelebaImgsHydraConf:

    force_extract: bool = False

    _target_: str = field(init=False, repr=False, default="gymnos.datasets.celeba_imgs.dataset.CelebaImgs")
