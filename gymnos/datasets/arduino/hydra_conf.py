#
#
#   Arduino Hydra conf
#
#

from dataclasses import dataclass, field


@dataclass
class ArduinoHydraConf:

    # TODO: add custom parameters

    _target_: str = field(init=False, repr=False, default="gymnos.datasets.arduino.dataset.Arduino")
