#
#
#   Mnist Imgs Hydra conf
#
#

from dataclasses import dataclass, field


@dataclass
class MnistImgsHydraConf:

    force_extract: bool = False

    _target_: str = field(init=False, repr=False, default="gymnos.datasets.mnist_imgs.dataset.MnistImgs")
