#
#
#   K-Best Transformer
#
#

from .preprocessor import Preprocessor
from sklearn.feature_selection import SelectKBest, chi2


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
    """

    def __init__(self, scorer, k=1000):
        if scorer == "chi2":
            score_func = chi2
        else:
            raise ValueError("Scorer {} not supported".format(scorer))

        self.kbest = SelectKBest(
            score_func=score_func,
            k=k
        )

    def fit(self, X, y=None):
        self.kbest.fit(X, y)
        return self

    def transform(self, X):
        return self.kbest.transform(X)
