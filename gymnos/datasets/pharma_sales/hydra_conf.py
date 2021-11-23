#
#
#   Pharma Sales Hydra conf
#
#

from dataclasses import dataclass, field


@dataclass
class PharmaSalesHydraConf:

    # TODO: add custom parameters

    _target_: str = field(init=False, repr=False, default="gymnos.datasets.pharma_sales.dataset.PharmaSales")
