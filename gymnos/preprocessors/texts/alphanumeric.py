#
#
#   Alphanumeric Preprocessor
#
#

import re

from ..preprocessor import Preprocessor
from ...utils.iterator_utils import apply


class Alphanumeric(Preprocessor):
    """
    Keep only alphanumeric characters.
    """

    def fit(self, X, y=None):
        return self

    def fit_generator(self, generator):
        return self

    def _transform_sample(self, x):
        pattern = r'([^\s\w]|_)+'
        return re.sub(pattern, " ", x)

    def transform(self, X):
        return apply(X, self._transform_sample)

    def save(self, save_dir):
        pass

    def restore(self, save_dir):
        pass
