#
#
#   Model
#
#

from .hydra_conf import A2CHydraConf

hydra_conf = A2CHydraConf

pip_dependencies = [
    "torch",
    "numpy",
    "torchvision"
]

apt_dependencies = []
