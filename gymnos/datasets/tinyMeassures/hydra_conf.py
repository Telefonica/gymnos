#
#
#   Tiny Meassures Hydra conf
#
#

from dataclasses import dataclass, field


@dataclass
class TinyMeassuresHydraConf:

    sensor_sample_rate: int = 15
    samples_per_take: int = 15
    sample_time: float = 0.06

    _target_: str = field(init=False, repr=False,
                          default="gymnos.datasets.tinyMeassures.dataset.TinyMeassures")
