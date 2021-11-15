#
#
#   Model
#
#

from .hydra_conf import NeuralNetworkClassifierHydraConf

hydra_conf = NeuralNetworkClassifierHydraConf

pip_dependencies = [
    'pandas',
    'os',
    'numpy',
    'sklearn',
    'tensorflow',
    'joblib'
]

apt_dependencies = []
