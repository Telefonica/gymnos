#
#
#   I/O Utils
#
#

import json
import numpy as np


def read_from_text(file_path):
    with open(file_path) as f:
        return f.read()


def read_from_json(file_path):
    with open(file_path) as f:
        return json.load(f)


def _json_default(o):
    # Hack to save numpy number with JSON module
    if isinstance(o, np.int32) or isinstance(o, np.int64):
        return int(o)
    if isinstance(o, np.float32) or isinstance(o, np.float64):
        return float(o)
    raise TypeError


def save_to_json(path, obj, indent=4):
    with open(path, "w") as outfile:
        json.dump(obj, outfile, indent=indent, default=_json_default)
