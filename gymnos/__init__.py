#
#
#   Gymnos
#
#

from . import utils, config
from .__about__ import __description__, __author__, __version__, __license__, __url__  # noqa: F401


GYMNOS_HOME = config.get_gymnos_home()
GYMNOS_CONFIG = config.get_gymnos_config()
GYMNOS_CONFIG_PATH = config.get_gymnos_config_path()
