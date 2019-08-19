import os

from ..utils.io_utils import import_from_json


def load(name, **params):
    try:
        Dataset = import_from_json(os.path.join(os.path.dirname(__file__), "..", "var", "datasets.json"), name)
    except KeyError as e:
        raise ValueError("Dataset with name {} not found".format(name)) from e
    return Dataset(**params)
