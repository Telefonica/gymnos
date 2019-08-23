#
#
#   Data Augmentors
#
#

from ..registration import ComponentRegistry


registry = ComponentRegistry("data augmentor")  # global component registry


def register(name, entry_point):
    """
    Register dataset.

    Parameters
    -----------
    name: str
        Dataset id to register
    entry_point: str
        Dataset path
    """
    return registry.register(name, entry_point)


def load(name, **kwargs):
    """
    Load registered dataset

    Parameters
    ----------
    name: str
        Dataset id to load
    **kwargs: any
        Dataset constructor arguments

    Returns
    --------
    dataset: gymnos.datasets.dataset.Dataset
        Dataset instance
    """
    return registry.load(name, **kwargs)
