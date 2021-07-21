#
#
#   Conf
#
#

from typing import Optional
from dataclasses import dataclass, field


@dataclass
class DogsVsCatsHydraConf:

    force_download: bool = False
    max_workers: Optional[int] = None

    _target_: str = field(init=False, default="gymnos.datasets.dogs_vs_cats.dataset.DogsVsCats")
