#
#
#   Iterator Utils
#
#

import numpy as np
import pandas as pd

from tqdm import tqdm


def apply(data, func, verbose=True):
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
            return np.array([func(row) for row in tqdm(data)])

        return np.apply_along_axis(func, axis=0, arr=data)
    else:
        return [func(row) for row in data]
