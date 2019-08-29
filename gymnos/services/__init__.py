from ..registration import ComponentRegistry

registry = ComponentRegistry("service")  # global component registry


def register(name, entry_point):
    """
    Register service.

    Parameters
    -----------
    name: str
        Service id to register
    entry_point: str
        Service path
    """
    return registry.register(name, entry_point)


def load(name, **kwargs):
    """
    Load registered service

    Parameters
    ----------
    name: str
        Service id to load
    **kwargs: any
        Service constructor arguments

    Returns
    --------
    service: gymnos.services.service.service
        Service instance
    """
    return registry.load(name, **kwargs)
