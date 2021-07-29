#
#
#   Model
#
#

from .hydra_conf import TransferEfficientNetHydraConf

hydra_conf = TransferEfficientNetHydraConf

requirements = [
    "numpy",
    "torch",
    "Pillow",
    "torchvision",
    "torchmetrics",
    "efficientnet_pytorch==0.7.0",
    "pytorch-lightning>=1.0.0"
]

packages = [
]
