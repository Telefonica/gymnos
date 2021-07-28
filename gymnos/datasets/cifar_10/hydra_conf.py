#
#
#   Cifar 10 Hydra conf
#
#

from dataclasses import dataclass, field


@dataclass
class CIFAR10HydraConf:

    # TODO: add custom parameters

    _target_: str = field(init=False, default="gymnos.datasets.cifar_10.dataset.CIFAR10")
