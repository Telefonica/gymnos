#
#
#   Data Augmentors
#
#

from ..registration import ComponentRegistry

# MARK: Public API
from .data_augmentor import DataAugmentor, Pipeline  # noqa: F401


registry = ComponentRegistry("data augmentor")  # global component registry


def register(type, entry_point):
    """
    Register data augmentor.

    Parameters
    -----------
    type: str
        Data augmentor id to register
    entry_point: str
        Data augmentor path
    """
    return registry.register(type, entry_point)


def load(*args, **kwargs):
    """
    Load registered data augmentor

    Parameters
    ----------
    type: str
        Data augmentor id to load
    **kwargs: any
        Data augmentor constructor arguments

    Returns
    --------
    data augmentor: gymnos.data augmentors.data augmentor.data augmentor
        Data augmentor instance
    """
    return registry.load(*args, **kwargs)
