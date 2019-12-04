#
#
#   Repetition XGBoost
#
#

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
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

    Note
    ----
    This model requires binary labels.
    """

    def __init__(self, cv=5, search=None):
        self.model = lazy_imports.xgboost.XGBClassifier(objective="binary:logistic", random_state=42, max_depth=3,
                                                        n_estimators=100)
        self.cv = cv
        self.search = search

    def fit(self, X, y):
        model_search = self.model

        if self.search == "grid_search":
            XGBOOST_GRID = {'Classifier__max_depth': [2, 4, 6],
                            'Classifier__n_estimators': np.geomspace(50, 500, num=5).astype(int)}
            model_search = GridSearchCV(estimator=model_search, param_grid=XGBOOST_GRID,
                                        scoring='roc_auc', refit=True, cv=self.cv, verbose=3)
        elif self.search == "random_search":
            XGBOOST_RANDOM_GRID = {'Classifier__max_depth': [2, 4, 6],
                                   'Classifier__n_estimators': np.geomspace(50, 500, num=5).astype(int)}
            model_search = RandomizedSearchCV(estimator=model_search, param_distributions=XGBOOST_RANDOM_GRID,
                                              scoring='roc_auc', cv=self.cv, refit=True,
                                              random_state=314, verbose=3)
        else:
            pass
        self.model = model_search.fit(X, y)
        if self.search in ["grid_search", "random_search"]:
            self.model = model_search.best_estimator_

    def evaluate(self, X, y):
        result = self.predict(X)
        cr = classification_report(y, result, output_dict=True)
        probs = self.predict_proba(X)
        auc = roc_auc_score(y, probs)
        return auc, cr

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
