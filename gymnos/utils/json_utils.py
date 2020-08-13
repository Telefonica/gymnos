#
#
#   JSON utils
#
#

import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder to automatically convert numPy arrays to lists.
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def default(o):
    """
    Hack to save numpy number with JSON module.
    The main difference with NumpyEncoder refers to how each one handle
    data types. NumpyEncoder will convert numPy arrays to list but it will
    keep numPy data types. This function will try to convert numPy datatypes
    to the closest Python data type.
    """
    if isinstance(o, np.int32) or isinstance(o, np.int64):
        return int(o)
    if isinstance(o, np.float32) or isinstance(o, np.float64):
        return float(o)
    raise TypeError


def read_json(file_path, *args, **kwargs):
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
        return json.load(f, *args, **kwargs)


def save_to_json(path, obj, indent=4, *args, **kwargs):
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
        json.dump(obj, outfile, indent=indent, cls=NumpyEncoder, default=default, *args, **kwargs)
