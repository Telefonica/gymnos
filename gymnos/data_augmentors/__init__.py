#
#
#   Data Augmentors
#
#

from ..registration import ComponentRegistry


registry = ComponentRegistry("data augmentor")  # global component registry


def register(name, entry_point):
    """
    Register data augmentor.

    Parameters
    -----------
    name: str
        Data augmentor id to register
    entry_point: str
        Data augmentor path
    """
    return registry.register(name, entry_point)


def load(name, **kwargs):
    """
    Load registered data augmentor

    Parameters
    ----------
    name: str
        Data augmentor id to load
    **kwargs: any
        Data augmentor constructor arguments

    Returns
    --------
    data augmentor: gymnos.data augmentors.data augmentor.data augmentor
        Data augmentor instance
    """
    return registry.load(name, **kwargs)
