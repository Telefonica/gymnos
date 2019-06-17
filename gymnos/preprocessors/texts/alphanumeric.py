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

    def __transform_sample(self, x):
        return re.sub(r'([^\s\w]|_)+', ' ', x)  # only alphanumeric


    def transform(self, X):
        return apply(X, self.__transform_sample)
