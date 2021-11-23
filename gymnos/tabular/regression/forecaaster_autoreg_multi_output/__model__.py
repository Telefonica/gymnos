#
#
#   Model
#
#

from .hydra_conf import ForecaasterAutoregMultiOutputHydraConf

hydra_conf = ForecaasterAutoregMultiOutputHydraConf

pip_dependencies = [
    'pandas',
    'os',
    'numpy',
    'scikit-learn',
    'random',
    'skforecast',
    'matplotlib',
    'joblib'
]

apt_dependencies = []
