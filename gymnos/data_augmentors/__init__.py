import os

from ..utils.io_utils import import_from_json


def load(name, **params):
    """
    Load data augmentor by name

    Parameters
    -------------
    name: str
        Data Augmentor name
    **params: any
        Any parameter for data augmentor constructor

    Returns
    ----------
    data_augmentor: DataAugmentor
    """
    try:
        DataAugmentor = import_from_json(os.path.join(os.path.dirname(__file__), "..", "var", "data_augmentors.json"),
                                         name)
    except KeyError as e:
        raise ValueError("DataAugmentor with name {} not found".format(name)) from e
    return DataAugmentor(**params)
