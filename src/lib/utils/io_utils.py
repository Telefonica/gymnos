#
#
#   I/O Utils
#
#

import json
import commentjson
import numpy as np

from pydoc import locate


def read_from_text(file_path):
    """
    Read file text.

    Parameters
    ----------
    file_path: str
        File path to read.

    Returns
    -------
    file_text: str
        Text.
    """
    with open(file_path) as f:
        return f.read()


def count_lines(file_path):
    """
    Count lines from a file

    Parameters
    ----------
    file_path: str
        File path to count lines

    Returns
    --------
    int
        Number of lines
    """
    with open(file_path, "r") as f:
        return sum(1 for line in f)


def read_from_json(file_path, with_comments_support=False):
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
        if with_comments_support:
            return commentjson.load(f)
        else:
            return json.load(f)


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


def import_from_json(json_path, key):
    """
    Import module from a JSON file.
    The JSON structure must be in the following format:
    {
        <key>: <module_path (e.g lib.core.model.Model)>
    }

    Parameters
    ----------
    json_path: str
        JSON file path
    key: str
        JSON key to read module path

    Returns
    -------
    object
        Imported object
    """
    objects_ids_to_modules  = read_from_json(json_path)
    object_loc = objects_ids_to_modules[key]
    return locate(object_loc)
