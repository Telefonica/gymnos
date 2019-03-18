#
#
#   Divide preprocessor
#
#

from .preprocessor import Preprocessor


class Divide(Preprocessor):

    def __init__(self, factor):
        self.factor = factor

    def transform(self, X):
        return X / self.factor
