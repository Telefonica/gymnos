#
#
#   Iterator utils
#
#

import numpy as np
import pandas as pd

from gymnos.utils.iterator_utils import apply


def test_apply():
    arr = np.random.randint(0, 10, 100)

    def square(x):
        return x**2

    new_arr = apply(arr, square, verbose=False)

    assert new_arr.shape == arr.shape
    assert np.array_equal(new_arr, arr**2)

    arr = np.random.randint(0, 10, (5, 6))

    new_arr = apply(arr, square, verbose=False)

    assert new_arr.shape == arr.shape
    assert np.array_equal(new_arr, arr**2)

    df = pd.DataFrame(arr)

    new_df = apply(df, np.mean, verbose=False)

    assert isinstance(new_df, pd.Series)
    assert np.array_equal(new_df, df.apply(np.mean, axis=0))

    df = pd.DataFrame(arr)

    new_df = apply(df, square, verbose=False)

    assert new_df.shape == df.shape
    assert isinstance(new_df, pd.DataFrame)
    assert np.array_equal(new_df, df**2)
