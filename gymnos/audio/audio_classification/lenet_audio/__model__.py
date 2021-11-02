#
#
#   Model
#
#

from .hydra_conf import LenetAudioHydraConf

hydra_conf = LenetAudioHydraConf

pip_dependencies = [
    "torch",
    "torchaudio",
    "scikit-learn",
    "numpy"
]

apt_dependencies = []
