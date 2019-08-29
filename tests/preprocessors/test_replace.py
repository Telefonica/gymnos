#
#
#   Test replace
#
#

import numpy as np
from gymnos.preprocessors.replace import Replace


def test_transform():
    replace = Replace(from_val=0, to_val=1)

    zeros = np.zeros([10, 5], dtype=np.int32)

    new_zeros = replace.transform(zeros)

    assert np.array_equal(new_zeros, np.ones_like(zeros))
