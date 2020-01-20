#
#
#   K-Best Transformer
#
#

import os
import dill

from .preprocessor import Preprocessor
from ..utils.lazy_imports import lazy_imports as lazy


class KBest(Preprocessor):
    """
    Select features according to the k highest scores.
    Based on `sklearn.feature_selection.SelectKBest <https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html>`_.

    Parameters
    ----------
    scorer: str
        Scoring function. The currently available scorers are the following:

            - ``"chi2"``: `sklearn.feature_selection.chi2 <https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html#sklearn.feature_selection.chi2>`_
    k: int, optional
        Number of top features to select. The “all” option bypasses selection, for use in a parameter search.
    """  # noqa: E501

    def __init__(self, scorer, k=1000):
        sklearn = __import__("{}.feature_selection".format(lazy.sklearn.__name__))

        if scorer == "chi2":
            score_func = sklearn.feature_selection.chi2
        else:
            raise ValueError("Scorer {} not supported".format(scorer))

        self.kbest = sklearn.feature_selection.SelectKBest(
            score_func=score_func,
            k=k
        )

    def fit(self, X, y=None):
        self.kbest.fit(X, y)
        return self

    def transform(self, X):
        return self.kbest.transform(X)

    def save(self, save_dir):
        with open(os.path.join(save_dir, "kbest.pkl"), "wb") as fp:
            dill.dump(self.kbest, fp)

    def restore(self, save_dir):
        with open(os.path.join(save_dir, "kbest.pkl"), "rb") as fp:
            self.kbest = dill.load(fp)
