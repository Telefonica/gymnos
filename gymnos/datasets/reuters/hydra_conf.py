#
#
#   Reuters Hydra conf
#
#

from dataclasses import dataclass, field


@dataclass
class ReutersHydraConf:

    # TODO: add custom parameters

    _target_: str = field(init=False, default="gymnos.datasets.reuters.Reuters")
