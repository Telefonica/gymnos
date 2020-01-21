#
#
#   Standard Scaler
#
#

import os
import dill

from .preprocessor import Preprocessor
from ..utils.lazy_imports import lazy_imports as lazy


class StandardScaler(Preprocessor):
    """
    Standardize features by removing the mean and scaling to unit variance

    Parameters
    -----------
    copy: bool, optional
        If False, try to avoid a copy and do inplace scaling instead
    with_mean: bool, optional
        If True, center the data before scaling.
    with_std: bool, optional
        If True, scale the data to unit variance.
    """

    def __init__(self, copy=True, with_mean=True, with_std=True):
        sklearn = __import__("{}.preprocessing".format(lazy.sklearn.__name__))

        self.std_scaler = sklearn.preprocessing.StandardScaler(copy=copy, with_std=with_std,
                                                               with_mean=with_mean)

    def fit(self, X, y=None):
        return self.std_scaler.fit(X, y)

    def fit_generator(self, generator):
        for X, y in generator:
            self.std_scaler.partial_fit(X, y)

        return self

    def transform(self, X):
        return self.std_scaler.transform(X)

    def save(self, save_dir):
        with open(os.path.join(save_dir, "std_scaler.pkl"), "wb") as fp:
            dill.dump(self.std_scaler, fp)

    def restore(self, save_dir):
        with open(os.path.join(save_dir, "std_scaler.pkl"), "rb") as fp:
            self.std_scaler = dill.load(fp)
