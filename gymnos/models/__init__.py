import os

from ..utils.io_utils import import_from_json


def load(name, **params):
    try:
        Model = import_from_json(os.path.join(os.path.dirname(__file__), "..", "var", "models.json"),
                                 name)
    except KeyError as e:
        raise ValueError("Model with name {} not found".format(name)) from e
    return Model(**params)
