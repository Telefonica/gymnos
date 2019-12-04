#
#
#   Repetition Random Forest
#
#

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
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

    Note
    ----
    This model requires binary labels.
    """

    def __init__(self, cv=5, search=None):
        self.model = RandomForestClassifier(n_estimators=500)
        self.cv = cv
        self.search = search

    def fit(self, X, y):
        model_search = self.model
        if self.search == "grid_search":
            RANDOM_FOREST_GRID = {'n_estimators': [200, 500],
                                  'max_features': ['auto', 'sqrt', 'log2'],
                                  'max_depth': [4, 5, 6, 7, 8],
                                  'criterion': ['gini', 'entropy']}
            model_search = GridSearchCV(self.model, RANDOM_FOREST_GRID, scoring='roc_auc', cv=self.cv, n_jobs=-1,
                                        verbose=3)
        elif self.search == "random_search":
            n_estimators = np.geomspace(10, 250, num=8).astype(int)
            max_features = ['auto', 'sqrt']
            max_depth = np.geomspace(10, 250, num=8).astype(int)
            min_samples_split = [2, 5, 10]
            min_samples_leaf = [1, 2, 4]
            bootstrap = [True, False]
            RANDOM_FOREST_RANDOM_GRID = {'n_estimators': n_estimators,
                                         'max_features': max_features,
                                         'max_depth': max_depth,
                                         'min_samples_split': min_samples_split,
                                         'min_samples_leaf': min_samples_leaf,
                                         'bootstrap': bootstrap}
            model_search = RandomizedSearchCV(estimator=self.model, scoring='roc_auc',
                                              param_distributions=RANDOM_FOREST_RANDOM_GRID, n_iter=100, cv=3,
                                              verbose=3,
                                              random_state=42, n_jobs=-1)
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
