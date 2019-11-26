#
#
#   JSON utils
#
#

import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder to automatically convert numPy arrays to lists.
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def default(o):
    """
    Hack to save numpy number with JSON module.
    The main difference with NumpyEncoder refers to how each one handle
    data types. NumpyEncoder will convert numPy arrays to list but it will
    keep numPy data types. This function will try to convert numPy datatypes
    to the closest Python data type.
    """
    if isinstance(o, np.int32) or isinstance(o, np.int64):
        return int(o)
    if isinstance(o, np.float32) or isinstance(o, np.float64):
        return float(o)
    raise TypeError
