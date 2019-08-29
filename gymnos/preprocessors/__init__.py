#
#
#   Preprocessors
#
#

from ..registration import ComponentRegistry


registry = ComponentRegistry("preprocessor")  # global component registry


def register(name, entry_point):
    """
    Register preprocessor.

    Parameters
    -----------
    name: str
        Preprocessor id to register
    entry_point: str
        Preprocessor path
    """
    return registry.register(name, entry_point)


def load(name, **kwargs):
    """
    Load registered preprocessor

    Parameters
    ----------
    name: str
        Preprocessor id to load
    **kwargs: any
        Preprocessor constructor arguments

    Returns
    --------
    preprocessor: gymnos.preprocessors.preprocessor.preprocessor
        Preprocessor instance
    """
    return registry.load(name, **kwargs)
