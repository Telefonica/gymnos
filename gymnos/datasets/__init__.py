#
#
#   Datasets
#
#

from ..registration import ComponentRegistry


registry = ComponentRegistry("dataset")  # global component registry


def register(name, entry_point):
    return registry.register(name, entry_point)


def load(name, **kwargs):
    return registry.load(name, **kwargs)
