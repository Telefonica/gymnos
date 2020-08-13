#
#
#   Trackers
#
#

# MARK: Public API
from .tracker import Tracker    # noqa: F401

from ..registration import ComponentRegistry


registry = ComponentRegistry("tracker")  # global component registry


def register(type, entry_point):
    """
    Register model.

    Parameters
    -----------
    type: str
        Model id to register
    entry_point: str
        Model path
    """
    return registry.register(type, entry_point)


def load(*args, **kwargs):
    """
    Load registered model

    Parameters
    ----------
    type: str
        Model id to load
    **kwargs: any
        Model constructor arguments

    Returns
    --------
    model: gymnos.models.model.model
        Model instance
    """
    return registry.load(*args, **kwargs)
