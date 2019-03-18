#
#
#   Normalize Preprocessor
#
#

import numpy as np

from .preprocessor import Preprocessor


class Normalize(Preprocessor):

    def transform(self, X):
        return np.mean(X) / np.std(X)
