#
#
#  Keras modules
#
#

import os
import json

from pydoc import locate


KERAS_MODULES_PATH = os.path.join(os.path.dirname(__file__), "..", "var", "keras_modules.json")


def import_keras_module(name, module_type):
    with open(KERAS_MODULES_PATH) as fp:
        data = json.load(fp)

    try:
        return locate(data[module_type][name])
    except KeyError as e:
        raise ValueError(("A keras component ({}) was specified but was " +
                          "not found in {}").format(name, module_type)) from e
