#
#
#   Config
#
#

import enum

from typing import Optional
from dataclasses import dataclass


class Device(enum.Enum):
    CPU = "CPU"
    GPU = "GPU"


@dataclass
class SOFIALauncherConfig:
    _target_: str = "hydra_plugins.sofia_launcher.launcher.SOFIALauncher"

    project_name: str = "???"
    ref: Optional[str] = None
    device: Device = Device.CPU
