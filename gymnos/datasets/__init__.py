#
#
#   Datasets
#
#

from ..registration import ComponentRegistry


registry = ComponentRegistry("dataset")  # global component registry


def register(type, entry_point):
    """
    Register dataset.

    Parameters
    -----------
    type: str
        Dataset id to register
    entry_point: str
        Dataset path
    """
    return registry.register(type, entry_point)


def load(type, **kwargs):
    """
    Load registered dataset

    Parameters
    ----------
    type: str
        Dataset id to load
    **kwargs: any
        Dataset constructor arguments

    Returns
    --------
    dataset: gymnos.datasets.dataset.dataset
        Dataset instance
    """
    return registry.load(type, **kwargs)
