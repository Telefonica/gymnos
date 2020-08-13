#
#
#   Datasets
#
#

# MARK: Public API
from .dataset import Dataset  # noqa: F401

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


def load(*args, **kwargs):
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
    return registry.load(*args, **kwargs)
