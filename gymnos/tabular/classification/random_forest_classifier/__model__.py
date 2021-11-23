#
#
#   Model
#
#

from .hydra_conf import RandomForestClassifierHydraConf

hydra_conf = RandomForestClassifierHydraConf

pip_dependencies = [
    'pandas',
    'os',
    'numpy',
    'sklearn',
    'joblib'
]

apt_dependencies = []
