#
#
#   Model
#
#

from .hydra_conf import WganGpHydraConf

hydra_conf = WganGpHydraConf

pip_dependencies = [
    "torch",
    "torchvision"
]

apt_dependencies = []
