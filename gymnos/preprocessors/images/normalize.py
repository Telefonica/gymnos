#
#
#   Normalize
#
#

from ..preprocessor import Preprocessor


class Normalize(Preprocessor):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def fit(self, X, y=None):
        return self

    def fit_generator(self, generator):
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def save(self, save_dir):
        pass

    def restore(self, save_dir):
        pass
