import os

from ..utils.io_utils import import_from_json


def load(name, **params):
    try:
        Preprocessor = import_from_json(os.path.join(os.path.dirname(__file__), "..", "var", "preprocessors.json"),
                                        name)
    except KeyError as e:
        raise ValueError("Preprocessor with name {} not found".format(name)) from e
    return Preprocessor(**params)
