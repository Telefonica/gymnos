#
#
#   I/O utils
#
#

import json


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
    from gymnos.utils.json_utils import NumpyEncoder, default

    with open(path, "w") as outfile:
        json.dump(obj, outfile, indent=indent, cls=NumpyEncoder, default=default, *args, **kwargs)
