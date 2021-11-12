#
#
#   Model
#
#

from .hydra_conf import SAGANHydraConf

hydra_conf = SAGANHydraConf

pip_dependencies = [
    "tqdm"
    "torch",
    "torchvision"
]

apt_dependencies = []
