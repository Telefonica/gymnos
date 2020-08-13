# MARK: Public API
from .service import Service  # noqa: F401
from .download_manager import DownloadManager  # noqa: F401

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


def load(*args, **kwargs):
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
    return registry.load(*args, **kwargs)
