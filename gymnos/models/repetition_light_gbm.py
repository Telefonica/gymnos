#
#
#   Repetition LightGBM
#
#

from scipy.stats import randint, uniform
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from .mixins import SklearnMixin
from .model import Model
from ..utils.lazy_imports import lazy_imports


class RepetitionLightGBM(SklearnMixin, Model):
    """
    LightGBM supervised model.

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
        self.model = lazy_imports.lightgbm.LGBMClassifier(max_depth=-1, random_state=314, silent=True, metric='None',
                                                          n_jobs=4,
                                                          n_estimators=5000)
        self.cv = cv
        self.search = search

    def fit(self, X, y):
        model_search = self.model

        if self.search == "grid_search":
            LIGHT_GBM_GRID = {'n_estimators': [1000, 1500, 2000, 2500],
                              'max_depth': [4, 5, 8, -1],
                              'num_leaves': [15, 31, 63, 127],
                              'subsample': [0.6, 0.7, 0.8, 1.0],
                              'colsample_bytree': [0.6, 0.7, 0.8, 1.0]}
            model_search = GridSearchCV(estimator=model_search, param_grid=LIGHT_GBM_GRID,
                                        scoring='roc_auc', cv=self.cv, refit=True,
                                        verbose=3)
        elif self.search == "random_search":
            LIGHT_GBM_RANDOM_GRID = {'n_estimators': randint(1000, 2500),
                                     'max_depth': [4, 5, 8, -1],
                                     'num_leaves': [15, 31, 63, 127],
                                     'subsample': uniform(0.6, 0.4),
                                     'colsample_bytree': uniform(0.6, 0.4)}
            # This parameter defines the number of HP points to be tested
            n_HP_points_to_test = 100
            model_search = RandomizedSearchCV(estimator=model_search, param_distributions=LIGHT_GBM_RANDOM_GRID,
                                              n_iter=n_HP_points_to_test, scoring='roc_auc', cv=self.cv, refit=True,
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
