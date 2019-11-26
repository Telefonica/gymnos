#
#
#   Datasets
#
#

from ..registration import ComponentRegistry

# MARK: Public API
from .execution_environment import ExecutionEnvironment  # noqa: F401


registry = ComponentRegistry("execution_environment")  # global component registry


def register(type, entry_point):
    """
    Register dataset.

    Parameters
    -----------
    type: str
        Execution environment id to register
    entry_point: str
        Execution environment path
    """
    return registry.register(type, entry_point)


def load(type, **kwargs):
    """
    Load registered execution environment

    Parameters
    ----------
    type: str
        Execution environment id to load
    **kwargs: any
        Execution environment constructor arguments

    Returns
    --------
    dataset: gymnos.execution_environments.execution_environment.ExecutionEnvironment
        Execution environment instance
    """
    return registry.load(type, **kwargs)
