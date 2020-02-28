#
#
#   Repetition XGBoost
#
#

import numpy as np

from .mixins import SklearnMixin
from .model import Model
from ..utils.lazy_imports import lazy_imports

xgboost = lazy_imports.xgboost
sklearn_model_selection = __import__(f"{lazy_imports.sklearn.__name__}.model_selection")


class RepetitionXGBoost(SklearnMixin, Model):
    """
    XGBoost supervised model.

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
        self.model = xgboost.XGBClassifier(objective="binary:logistic", random_state=42, max_depth=3,
                                           n_estimators=100)
        self.cv = cv
        self.search = search
        self.scoring = scoring
        self.n_iter = n_iter

    def fit(self, x, y, validation_split=0, cross_validation=None):
        metrics = {}
        x = np.array(x)
        y = np.array(y)

        # create cross validation iterator
        cv = sklearn_model_selection.model_selection.ShuffleSplit(n_splits=self.cv, test_size=0.2, random_state=0)

        if self.search == "grid_search":
            xgboost_grid = {'Classifier__max_depth': [2, 4, 6],
                            'Classifier__n_estimators': np.geomspace(50, 500, num=5).astype(int)}
            self.model = sklearn_model_selection.model_selection.GridSearchCV(estimator=self.model,
                                                                              param_grid=xgboost_grid,
                                                                              scoring=self.scoring, refit=True, cv=cv,
                                                                              verbose=3,
                                                                              n_jobs=-1)
        elif self.search == "random_search":
            xgboost_random_grid = {'Classifier__max_depth': [2, 4, 6],
                                   'Classifier__n_estimators': np.geomspace(50, 500, num=5).astype(int)}
            self.model = sklearn_model_selection.model_selection.RandomizedSearchCV(
                estimator=self.model,
                param_distributions=xgboost_random_grid,
                scoring=self.scoring, cv=cv,
                refit=True,
                random_state=14, verbose=3,
                n_iter=self.n_iter,
                n_jobs=-1)
        else:
            pass

        self.model.fit(x, y)

        if self.search in ["grid_search", "random_search"]:
            metrics[self.scoring] = self.model.best_score_
        return metrics
