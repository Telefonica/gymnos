#
#
#   Repetition XGBoost
#
#

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from .mixins import SklearnMixin
from .model import Model
from ..utils.lazy_imports import lazy_imports


class RepetitionXGBoost(SklearnMixin, Model):
    """
    XGBoost supervised model.

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

    def __init__(self, cv=5, search=None, scoring='roc_auc'):
        self.model = lazy_imports.xgboost.XGBClassifier(objective="binary:logistic", random_state=42, max_depth=3,
                                                        n_estimators=100)
        self.cv = cv
        self.search = search
        self.scoring = scoring

    def fit(self, x, y, validation_split=0, cross_validation=None):
        model_search = self.model
        metrics = {}

        if self.search == "grid_search":
            xgboost_grid = {'Classifier__max_depth': [2, 4, 6],
                            'Classifier__n_estimators': np.geomspace(50, 500, num=5).astype(int)}
            model_search = GridSearchCV(estimator=model_search, param_grid=xgboost_grid,
                                        scoring=self.scoring, refit=True, cv=self.cv, verbose=3)
        elif self.search == "random_search":
            xgboost_random_grid = {'Classifier__max_depth': [2, 4, 6],
                                   'Classifier__n_estimators': np.geomspace(50, 500, num=5).astype(int)}
            model_search = RandomizedSearchCV(estimator=model_search, param_distributions=xgboost_random_grid,
                                              scoring=self.scoring, cv=self.cv, refit=True,
                                              random_state=314, verbose=3)
        else:
            pass
        self.model = model_search.fit(x, y)
        if self.search in ["grid_search", "random_search"]:
            self.model = model_search.best_estimator_
            metrics[self.scoring] = model_search.best_score_
        return metrics
