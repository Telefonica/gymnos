#
#
#   I/O Utils
#
#

import json
from pydoc import locate


def read_file_text(file_path):
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
    with open(json_path) as f:
        objects_ids_to_modules = json.load(f)
    object_loc = objects_ids_to_modules[key]
    return locate(object_loc)

