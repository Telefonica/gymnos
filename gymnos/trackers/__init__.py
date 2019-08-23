#
#
#   Trackers
#
#

from ..registration import ComponentRegistry


registry = ComponentRegistry("tracker")  # global component registry


def register(name, entry_point):
    return registry.register(name, entry_point)


def load(name, **kwargs):
    return registry.load(name, **kwargs)
