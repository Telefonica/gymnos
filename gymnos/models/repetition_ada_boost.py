#
#
#   Repetition AdaBoost
#
#

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ShuffleSplit

from .mixins import SklearnMixin
from .model import Model


class RepetitionAdaBoost(SklearnMixin, Model):
    """
    AdaBoost supervised model.

    Parameters
    ----------
    cv: int
        Number of chunks in cross validation
    search: str
        Type of hyperparameters search (grid search or random search)
    scoring: str
        Type of scoring to do the hyperparameters searching (such as 'auc_roc', 'recall',...).
    n_iter: int,
        Number of iterations of the searching. Valid only in if search=random search.

    Note
    ----
    This model requires binary labels.
    """

    def __init__(self, cv=5, search=None, scoring='roc_auc', n_iter=100):
        self.model = AdaBoostClassifier()
        self.cv = cv
        self.search = search
        self.scoring = scoring
        self.n_iter = n_iter

    def fit(self, x, y, validation_split=0, cross_validation=None):
        metrics = {}
        x = np.array(x)
        y = np.array(y)

        # create cross validation iterator
        cv = ShuffleSplit(n_splits=self.cv, test_size=0.2, random_state=0)

        if self.search == "grid_search":
            ada_boost_grid = {'n_estimators': [500, 1000, 2000], 'learning_rate': [.001, 0.01, .1]}
            self.model = GridSearchCV(estimator=self.model, param_grid=ada_boost_grid, scoring=self.scoring,
                                      refit=True, cv=cv, verbose=3, n_jobs=1)

        elif self.search == "random_search":
            ada_boost_random_grid = {'n_estimators': [500, 1000, 2000], 'learning_rate': [.001, 0.01, .1]}
            self.model = RandomizedSearchCV(estimator=self.model, param_distributions=ada_boost_random_grid,
                                            scoring=self.scoring, cv=cv, refit=True,
                                            random_state=14, verbose=3, n_jobs=-1, n_iter=self.n_iter)

        else:
            self.model.fit(x, y)

        metrics['search'] = self.model
        if self.search in ["grid_search", "random_search"]:
            metrics[self.scoring] = self.model.best_score_
        return metrics
