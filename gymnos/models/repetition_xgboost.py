#
#
#   Repetition AdaBoost
#
#

import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from .mixins import SklearnMixin
from .model import Model
from .utils.repetition_grids import XGBOOST_RANDOM_GRID, XGBOOST_GRID


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

    def __init__(self, cv, search):
        self.model = xgb.XGBClassifier(objective="binary:logistic", random_state=42, max_depth=3, n_estimators=100)
        self.cv = cv
        self.search = search

    def fit(self, X, y):
        model_search = self.model

        if self.search == "grid_search":
            model_search = GridSearchCV(estimator=model_search, param_grid=XGBOOST_GRID,
                                        scoring='roc_auc', refit=True, cv=self.cv, verbose=3)
        elif self.search == "random_search":
            model_search = RandomizedSearchCV(estimator=model_search, param_distributions=XGBOOST_RANDOM_GRID,
                                              scoring='roc_auc', cv=self.cv, refit=True,
                                              random_state=314, verbose=3)
        else:
            pass
        model_search.fit(X, y)
        if self.search in ["grid_search", "random_search"]:
            self.model = model_search.best_estimator_

    def fit_generator(self, generator):
        return {}

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        result = self.predict(X)
        cr = classification_report(y, result, output_dict=True)
        probs = self.model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, probs)
        auc = roc_auc_score(y, probs)
        return auc, cr, y, probs
