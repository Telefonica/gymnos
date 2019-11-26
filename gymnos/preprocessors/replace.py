#
#
#   Replace Preprocessor
#
#

import numpy as np

from .preprocessor import Preprocessor


class Replace(Preprocessor):
    """
    Replace value.

    Parameters
    ----------

    from_val: any
        Original value
    to_val: any
        New value
    """

    def __init__(self, from_val, to_val):
        self.from_val = from_val
        self.to_val = to_val

    def fit(self, X, y=None):
        return self

    def fit_generator(self, generator):
        return self

    def transform(self, X):
        return np.where(X == self.from_val, self.to_val, X)

    def save(self, save_dir):
        pass

    def restore(self, save_dir):
        pass
