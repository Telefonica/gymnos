#
#
#   Test divide
#
#

import numpy as np

from gymnos.preprocessors.divide import Divide


def test_transform():
    divide = Divide(factor=2)

    a = np.arange(10)

    new_a = divide.transform(a)

    assert np.array_equal(new_a, a / 2)
