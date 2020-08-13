#
#
#   Preprocessors
#
#

from ..registration import ComponentRegistry

# MARK: Public API
from .preprocessor import Preprocessor, Pipeline  # noqa: F401


registry = ComponentRegistry("preprocessor")  # global component registry


def register(type, entry_point):
    """
    Register preprocessor.

    Parameters
    -----------
    type: str
        Preprocessor id to register
    entry_point: str
        Preprocessor path
    """
    return registry.register(type, entry_point)


def load(*args, **kwargs):
    """
    Load registered preprocessor

    Parameters
    ----------
    type: str
        Preprocessor id to load
    **kwargs: any
        Preprocessor constructor arguments

    Returns
    --------
    preprocessor: gymnos.preprocessors.preprocessor.preprocessor
        Preprocessor instance
    """
    return registry.load(*args, **kwargs)
