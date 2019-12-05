#
#
#   Repetition Random Forest
#
#

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

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
        Type of hyperparameters search (grid search or random search)
    scoring: str
        Type of scoring to do the hyperparameter searching (such as 'auc_roc', 'recall',...)

    Note
    ----
    This model requires binary labels.
    """

    def __init__(self, cv=5, search=None, scoring='auc'):
        self.model = RandomForestClassifier(n_estimators=500)
        self.cv = cv
        self.search = search
        self.scoring = scoring

    def fit(self, x, y, validation_split=0, cross_validation=None):
        model_search = self.model
        metrics = {}

        if self.search == "grid_search":
            ranodm_forest_grid = {'n_estimators': [200, 500],
                                  'max_features': ['auto', 'sqrt', 'log2'],
                                  'max_depth': [4, 5, 6, 7, 8],
                                  'criterion': ['gini', 'entropy']}
            model_search = GridSearchCV(self.model, ranodm_forest_grid, scoring=self.scoring, cv=self.cv, n_jobs=-1,
                                        verbose=3)
        elif self.search == "random_search":
            n_estimators = np.geomspace(10, 250, num=8).astype(int)
            max_features = ['auto', 'sqrt']
            max_depth = np.geomspace(10, 250, num=8).astype(int)
            min_samples_split = [2, 5, 10]
            min_samples_leaf = [1, 2, 4]
            bootstrap = [True, False]
            ranodm_forest_random_grid = {'n_estimators': n_estimators,
                                         'max_features': max_features,
                                         'max_depth': max_depth,
                                         'min_samples_split': min_samples_split,
                                         'min_samples_leaf': min_samples_leaf,
                                         'bootstrap': bootstrap}
            model_search = RandomizedSearchCV(estimator=self.model, scoring=self.scoring,
                                              param_distributions=ranodm_forest_random_grid, n_iter=100, cv=3,
                                              verbose=3,
                                              random_state=42, n_jobs=-1)
        else:
            pass
        self.model = model_search.fit(x, y)
        if self.search in ["grid_search", "random_search"]:
            self.model = model_search.best_estimator_
        metrics[self.scoring] = model_search.best_score_
        return metrics
