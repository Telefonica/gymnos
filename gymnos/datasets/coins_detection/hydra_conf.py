#
#
#   Coins Detection Hydra conf
#
#

from dataclasses import dataclass, field


@dataclass
class CoinsDetectionHydraConf:

    # TODO: add custom parameters

    _target_: str = field(init=False, default="gymnos.datasets.coins_detection.dataset.CoinsDetection")
