#
#
#   Iterator Utils
#
#

import scipy
import numpy as np
import pandas as pd

from tqdm import tqdm


def count(data):
    if scipy.sparse.issparse(data):
        return data.shape[0]
    else:
        return len(data)


def apply(data, func, verbose=True):
    if isinstance(data, (pd.Series, pd.DataFrame)):
        if verbose:
            tqdm.pandas()
            return data.progress_apply(func)

        return data.apply(func)
    elif isinstance(data, np.ndarray):
        if verbose:
            return np.array([func(row) for row in tqdm(data)])

        return np.apply_along_axis(func, axis=0, arr=data)
    else:
        return [func(row) for row in data]
