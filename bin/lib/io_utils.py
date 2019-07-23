#
#
#   JSON
#
#

import json
import numpy as np


def read_json(file_path, object_pairs_hook=None):
    """
    Read JSON

    Parameters
    ----------
    file_path: str
        JSON file path

    Returns
    -------
    json: dict
        JSON data.
    """
    with open(file_path) as f:
        return json.load(f, object_pairs_hook=object_pairs_hook)


def _json_default(o):
    # Hack to save numpy number with JSON module
    if isinstance(o, np.int32) or isinstance(o, np.int64):
        return int(o)
    if isinstance(o, np.float32) or isinstance(o, np.float64):
        return float(o)
    raise TypeError


def save_to_json(path, obj, indent=4):
    """
    Save data to JSON file.

    Parameters
    ----------
    path: str
        JSON file path.
    obj: dict or list
        Object to save
    indent: int, optional
        Indentation to save file (pretty print JSON)
    """
    with open(path, "w") as outfile:
        json.dump(obj, outfile, indent=indent, default=_json_default)
