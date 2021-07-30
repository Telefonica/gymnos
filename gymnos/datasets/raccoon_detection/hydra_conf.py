#
#
#   Raccoon Detection Hydra conf
#
#

from dataclasses import dataclass, field


@dataclass
class RaccoonDetectionHydraConf:

    # TODO: add custom parameters

    _target_: str = field(init=False, default="gymnos.datasets.raccoon_detection.dataset.RaccoonDetection")
