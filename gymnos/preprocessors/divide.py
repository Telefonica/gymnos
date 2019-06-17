#
#
#   Divide preprocessor
#
#

from .preprocessor import Preprocessor


class Divide(Preprocessor):
    """
    Divide features by a factor.

    Parameters
    ----------
    factor: int or float
        Factor to divide
    """

    def __init__(self, factor):
        self.factor = factor

    def fit(self, X, y=None):
        return self

    def fit_generator(self, generator):
        return self

    def transform(self, X):
        return X / self.factor
