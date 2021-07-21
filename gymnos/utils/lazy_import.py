#
#
#   Lazy import
#
#

import lazy_object_proxy

from pydoc import locate


def lazy_import(entrypoint):
    def import_entrypoint():
        return locate(entrypoint)
    return lazy_object_proxy.Proxy(import_entrypoint)
