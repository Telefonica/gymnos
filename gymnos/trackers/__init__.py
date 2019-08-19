import os

from ..utils.io_utils import import_from_json


def load(name, **params):
    try:
        Tracker = import_from_json(os.path.join(os.path.dirname(__file__), "..", "var", "trackers.json"), name)
    except KeyError as e:
        raise ValueError("Tracker with name {} not found".format(name)) from e
    return Tracker(**params)
