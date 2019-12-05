#
#
#   Repetition SVC
#
#

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
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
        Type of hyperparameters search (grid search or random search)
    scoring: str
        Type of scoring to do the hyperparameter searching (such as 'auc_roc', 'recall',...)

    Note
    ----
    This model requires binary labels.
    """

    def __init__(self, cv=5, search=None, scoring='auc'):
        self.model = SVC(probability=True)
        self.cv = cv
        self.search = search
        self.scoring = scoring

    def fit(self, x, y, validation_split=0, cross_validation=None):
        model_search = self.model
        metrics = {}

        if self.search == "grid_search":
            svc_grid = {'kernel': ('linear', 'rbf'),
                        'C': (1, 0.25, 0.5, 0.75),
                        'gamma': (1, 2, 3, 'auto'),
                        'decision_function_shape': ('ovo', 'ovr'),
                        'shrinking': (True, False)}
            model_search = GridSearchCV(estimator=model_search, param_grid=svc_grid,
                                        scoring=self.scoring, refit=True, cv=self.cv, verbose=3)
        elif self.search == "random_search":
            svc_random_grid = {'kernel': ('linear', 'rbf'),
                               'C': np.geomspace(0.1, 4, num=8),
                               'gamma': (1, 2, 3, 'auto'),
                               'decision_function_shape': ('ovo', 'ovr'),
                               'shrinking': (True, False)}
            model_search = RandomizedSearchCV(estimator=model_search, param_distributions=svc_random_grid,
                                              scoring=self.scoring, cv=self.cv, refit=True,
                                              random_state=314, verbose=3)
        else:
            pass
        self.model = model_search.fit(x, y)
        if self.search in ["grid_search", "random_search"]:
            self.model = model_search.best_estimator_
        metrics[self.scoring] = model_search.best_score_
        return metrics
