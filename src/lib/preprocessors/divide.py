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

    def transform(self, X):
        return X / self.factor
