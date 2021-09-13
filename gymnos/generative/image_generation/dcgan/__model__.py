#
#
#   Model
#
#

from .hydra_conf import DCGANHydraConf

hydra_conf = DCGANHydraConf

pip_dependencies = [
    "Pillow",
    "tqdm",
    "torch",
    "torchvision"
]

apt_dependencies = []
