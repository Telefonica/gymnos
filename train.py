#
#
#   Train
#   The shell entry point `$ gymnos-train` is also available
#
#

import hydra
from omegaconf import DictConfig

from gymnos.cli.train import main


@hydra.main(config_path="./conf", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    return main(cfg)


if __name__ == "__main__":
    hydra_entry()
