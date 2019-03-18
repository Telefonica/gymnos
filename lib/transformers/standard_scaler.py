#
#
#   Standard Scaler
#
#

import sklearn

from sklearn.base import TransformerMixin


class StandardScaler(TransformerMixin):

    def __init__(self):
        self.scaler = sklearn.preprocessing.StandardScaler()

    def fit(self, X, y=None):
        raise NotImplementedError()

    def transform(self, X, y=None):
        raise NotImplementedError()
