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
    device: Device
    ref: Optional[str] = None
    verbose: bool = True
    notify_on_completion: bool = False

    _target_: str = field(init=False, repr=False, default="hydra_plugins.sofia_launcher.launcher.SOFIALauncher")
