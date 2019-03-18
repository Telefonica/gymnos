#
#
#   K-Best Transformer
#
#

from .transformer import Transformer

from sklearn.feature_selection import SelectKBest, chi2


class KBest(Transformer):

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

    def transform(self, X, y=None):
        return self.kbest.transform(X, y)
