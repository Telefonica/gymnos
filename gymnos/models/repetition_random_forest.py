#
#
#   Repetition Random Forest
#
#

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ShuffleSplit

from .mixins import SklearnMixin
from .model import Model


class RepetitionRandomForest(SklearnMixin, Model):
    """
    Random Forest supervised model.

    Parameters
    ----------
    cv: int
        Number of chunks in cross validation
    search: str
        Type of hyperparameters search (grid search or random search).
    scoring: str
        Type of scoring to do the hyperparameters searching (such as 'auc_roc', 'recall',...).
    n_iter: int,
        Number of iterations of the searching. Valid only in if search=random search.

    Note
    ----
    This model requires binary labels.
    """

    def __init__(self, cv=5, search=None, scoring='roc_auc', n_iter=100):
        self.model = RandomForestClassifier(n_estimators=500)
        self.cv = cv
        self.search = search
        self.scoring = scoring
        self.n_iter = n_iter

    def fit(self, x, y, validation_split=0, cross_validation=None):
        metrics = {}

        # create cross validation iterator
        cv = ShuffleSplit(n_splits=self.cv, test_size=0.2, random_state=0)

        if self.search == "grid_search":
            random_forest_grid = {'n_estimators': [200, 500],
                                  'max_features': ['auto', 'sqrt', 'log2'],
                                  'max_depth': [4, 5, 6, 7, 8],
                                  'criterion': ['gini', 'entropy']}
            self.model = GridSearchCV(self.model, random_forest_grid, refit=True, scoring=self.scoring,
                                      cv=cv, n_jobs=-1, verbose=3)
        elif self.search == "random_search":
            n_estimators = np.geomspace(10, 250, num=8).astype(int)
            max_features = ['auto', 'sqrt']
            max_depth = np.geomspace(10, 250, num=8).astype(int)
            min_samples_split = [2, 5, 10]
            min_samples_leaf = [1, 2, 4]
            bootstrap = [True, False]
            random_forest_random_grid = {'n_estimators': n_estimators,
                                         'max_features': max_features,
                                         'max_depth': max_depth,
                                         'min_samples_split': min_samples_split,
                                         'min_samples_leaf': min_samples_leaf,
                                         'bootstrap': bootstrap}
            self.model = RandomizedSearchCV(estimator=self.model, param_distributions=random_forest_random_grid,
                                            scoring=self.scoring, cv=cv, refit=True,
                                            random_state=14, verbose=3, n_jobs=-1, n_iter=self.n_iter)
        else:
            pass
        self.model.fit(x, y)
        if self.search in ["grid_search", "random_search"]:
            metrics[self.scoring] = self.model.best_score_
            metrics["best_params"] = self.model.best_params_
        return metrics
