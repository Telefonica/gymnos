#
#
#   Models
#
#

# MARK: Public API
from .model import Model   # noqa: F401

from ..registration import ComponentRegistry


registry = ComponentRegistry("model")  # global component registry


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


def load(type, **kwargs):
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
    return registry.load(type, **kwargs)
