#
#
#   Repetition SVC
#
#

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ShuffleSplit
from sklearn.svm import SVC

from .mixins import SklearnMixin
from .model import Model


class RepetitionSVC(SklearnMixin, Model):
    """
    SVC supervised model.

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
        self.model = SVC(probability=True)
        self.cv = cv
        self.search = search
        self.scoring = scoring
        self.n_iter = n_iter
        self.model_search = None

    def fit(self, x, y, validation_split=0, cross_validation=None):
        metrics = {}

        # create cross validation iterator
        cv = ShuffleSplit(n_splits=self.cv, test_size=0.2, random_state=0)

        if self.search == "grid_search":
            svc_grid = {'kernel': ('linear', 'rbf'),
                        'C': (1, 0.25, 0.5, 0.75),
                        'gamma': (1, 2, 3, 'auto'),
                        'decision_function_shape': ('ovo', 'ovr'),
                        'shrinking': (True, False)}
            self.model_search = GridSearchCV(estimator=self.model, param_grid=svc_grid,
                                             scoring=self.scoring, refit=True, cv=cv, verbose=3, n_jobs=-1)
            self.model_search.fit(x, y)
            self.model = self.model_search.best_estimator_
        elif self.search == "random_search":
            svc_random_grid = {'kernel': ('linear', 'rbf'),
                               'C': np.geomspace(0.1, 4, num=8),
                               'gamma': (1, 2, 3, 'auto'),
                               'decision_function_shape': ('ovo', 'ovr'),
                               'shrinking': (True, False)}
            self.model_search = RandomizedSearchCV(estimator=self.model, param_distributions=svc_random_grid,
                                                   scoring=self.scoring, cv=cv, refit=True,
                                                   random_state=14, verbose=3, n_jobs=-1, n_iter=self.n_iter)
            self.model_search.fit(x, y)
            self.model = self.model_search.best_estimator_
        else:
            self.model.fit(x, y)
            self.model_search = self.model

        metrics['search'] = self.model_search
        if self.search in ["grid_search", "random_search"]:
            metrics[self.scoring] = self.model_search.best_score_
            metrics["best_params"] = self.model_search.best_params_
            metrics['search'] = self.model_search
        return metrics