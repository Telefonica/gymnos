#
#
#   Hello Dataset Hydra conf
#
#

from dataclasses import dataclass, field


@dataclass
class HelloDatasetHydraConf:

    # TODO: add custom parameters

    _target_: str = field(init=False, repr=False, default="gymnos.datasets.hello_dataset.dataset.HelloDataset")
