#
#
#   Config
#
#

import enum

from typing import Optional
from dataclasses import dataclass, field


class Device(str, enum.Enum):
    CPU = "CPU"
    GPU = "GPU"


@dataclass
class SOFIALauncherHydraConf:

    project_name: str
    ref: Optional[str] = None
    device: Device = Device.CPU
    verbose: bool = True

    _target_: str = field(init=False, repr=False, default="hydra_plugins.sofia_launcher.launcher.SOFIALauncher")
