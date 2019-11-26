#
#
#   Iterator Utils
#
#

import numpy as np
import pandas as pd

from tqdm import tqdm


def apply(data, func, verbose=False):
    """
    Apply function to data, optionally showing a progress bar. Function is applied to rows.

    Parameters
    ----------
    data: array_like
        Data to apply function
    func: function
        Function to apply to data.
    verbose: bool, optional
        Whether or not show a progress bar

    Returns
    -------
    new_data: array_like
        Data with applied function.
    """
    if isinstance(data, (pd.Series, pd.DataFrame)):
        if verbose:
            tqdm.pandas()
            return data.progress_apply(func)

        return data.apply(func)
    elif isinstance(data, np.ndarray):
        if verbose:
            data = tqdm(data)

        return np.array([func(row) for row in data])
    else:
        if verbose:
            data = tqdm(data)

        return [func(row) for row in data]
