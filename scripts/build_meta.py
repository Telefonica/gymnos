#
#
#   Build
#
#

import argparse

from omegaconf import OmegaConf
from gymnos.cli.utils import build_meta

parser = argparse.ArgumentParser()
parser.add_argument("path")

args = parser.parse_args()

meta = build_meta()

with open(args.path, "w") as fp:
    OmegaConf.save(meta, fp)
