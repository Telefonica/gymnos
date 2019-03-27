#
#
#   Replace Preprocessor
#
#

import numpy as np

from .preprocessor import Preprocessor


class Replace(Preprocessor):

    def __init__(self, from_val, to_val):
        self.from_val = from_val
        self.to_val = to_val

    def transform(self, x):
        return np.where(x == self.from_val, self.to_val, x)
