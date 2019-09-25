from ..registration import ComponentRegistry

registry = ComponentRegistry("service")  # global component registry


def register(type, entry_point):
    """
    Register service.

    Parameters
    -----------
    type: str
        Service id to register
    entry_point: str
        Service path
    """
    return registry.register(type, entry_point)


def load(type, **kwargs):
    """
    Load registered service

    Parameters
    ----------
    type: str
        Service id to load
    **kwargs: any
        Service constructor arguments

    Returns
    --------
    service: gymnos.services.service.service
        Service instance
    """
    return registry.load(type, **kwargs)
