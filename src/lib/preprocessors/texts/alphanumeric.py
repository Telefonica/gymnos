#
#
#   Alphanumeric Preprocessor
#
#

import re

from ..preprocessor import Preprocessor
from ...utils.iterator_utils import apply


class Alphanumeric(Preprocessor):

    def __transform_sample(self, x):
        return re.sub(r'([^\s\w]|_)+', ' ', x)  # only alphanumeric


    def transform(self, X):
        return apply(X, self.__transform_sample)
