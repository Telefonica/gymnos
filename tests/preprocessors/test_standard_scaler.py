#
#
#   Test standard scaler
#
#

import numpy as np

from gymnos.preprocessors.standard_scaler import StandardScaler


def test_transform():
    std_scaler = StandardScaler()
    a = np.arange(9).reshape([3, 3])
    std_scaler.fit(a)

    new_a = std_scaler.transform(a)

    mean = a.mean(axis=0)
    std = a.std(axis=0)

    assert np.array_equal(new_a, (a - mean) / std)
