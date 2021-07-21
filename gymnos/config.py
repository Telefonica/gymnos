#
#
#   Config
#
#

import os
import json

from typing import Optional
from dacite import from_dict
from dataclasses import field, asdict, dataclass


def get_gymnos_home():
    return os.getenv("GYMNOS_HOME", os.path.expanduser(os.path.join("~", ".gymnos")))


def get_gymnos_config_path():
    return os.path.join(get_gymnos_home(), "gymnos.json")


@dataclass
class SOFIAConfig:
    access_token: Optional[str] = None


@dataclass
class Config:
    sofia: SOFIAConfig = field(default_factory=SOFIAConfig)


def get_gymnos_config() -> Config:
    config_path = get_gymnos_config_path()
    if not os.path.isfile(config_path):
        set_gymnos_config(Config())

    with open(config_path) as fp:
        data = json.load(fp)

    return from_dict(data_class=Config, data=data)


def set_gymnos_config(config: Config):
    config_path = get_gymnos_config_path()

    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, "w") as fp:
        json.dump(asdict(config), fp)
