#
#
#   Trackers
#
#

from ..registration import ComponentRegistry


registry = ComponentRegistry("tracker")  # global component registry


def register(name, entry_point):
    """
    Register model.

    Parameters
    -----------
    name: str
        Model id to register
    entry_point: str
        Model path
    """
    return registry.register(name, entry_point)


def load(name, **kwargs):
    """
    Load registered model

    Parameters
    ----------
    name: str
        Model id to load
    **kwargs: any
        Model constructor arguments

    Returns
    --------
    model: gymnos.models.model.model
        Model instance
    """
    return registry.load(name, **kwargs)
