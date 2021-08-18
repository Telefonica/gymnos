#
#
#   Build
#
#

import argparse

from omegaconf import OmegaConf
from gymnos.utils import remove_prefix
from gymnos.cli.utils import iter_modules

parser = argparse.ArgumentParser()
parser.add_argument("path")

args = parser.parse_args()

meta = OmegaConf.create()

meta["models"] = OmegaConf.create()
meta["envs"] = OmegaConf.create()
meta["datasets"] = OmegaConf.create()

for module in iter_modules("__model__.py"):
    modname = remove_prefix(module.__package__, "gymnos.")
    meta["models"][modname] = OmegaConf.structured(getattr(module, "hydra_conf"))

# Register datasets for Hydra
for module in iter_modules("__dataset__.py"):
    *_, modname = module.__package__.split(".")
    meta["datasets"][modname] = OmegaConf.structured(getattr(module, "hydra_conf"))

# Register envs for Hydra
for module in iter_modules("__env__.py"):
    *_, modname = module.__package__.split(".")
    meta["envs"][modname] = OmegaConf.structured(getattr(module, "hydra_conf"))

with open(args.path, "w") as fp:
    OmegaConf.save(meta, fp)
