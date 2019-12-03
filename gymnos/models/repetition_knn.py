#
#
#   Repetition KNN
#
#

from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier

from .mixins import SklearnMixin
from .model import Model


class RepetitionKNN(SklearnMixin, Model):
    """
    KNN supervised model.

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
        self.model = KNeighborsClassifier(n_neighbors=5)
        self.cv = cv
        self.search = search

    def fit(self, X, y):
        model_search = self.model
        k_range = list(range(1, 31))
        weight_options = ['uniform', 'distance']

        if self.search == "grid_search":
            KNN_GRID = {'n_neighbors': k_range,
                        'weights': weight_options}
            model_search = GridSearchCV(estimator=model_search, param_grid=KNN_GRID,
                                        scoring='roc_auc', refit=True, cv=self.cv, verbose=3)
        elif self.search == "random_search":
            KNN_RANDOM_GRID = {'n_neighbors': k_range,
                               'weights': weight_options}
            model_search = RandomizedSearchCV(estimator=model_search, param_distributions=KNN_RANDOM_GRID,
                                              scoring='roc_auc', cv=self.cv, refit=True,
                                              random_state=314, verbose=3)
        else:
            pass
        self.fitted_model_ = model_search.fit(X, y)
        if self.search in ["grid_search", "random_search"]:
            self.fitted_model_ = model_search.best_estimator_

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        result = self.predict(X)
        cr = classification_report(y, result, output_dict=True)
        probs = self.predict_proba(X)
        auc = roc_auc_score(y, probs)
        return auc, cr

    def predict_proba(self, X):
        return self.fitted_model_.predict_proba(X)[:, 1]
