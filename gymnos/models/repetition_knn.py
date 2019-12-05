#
#
#   Repetition KNN
#
#

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
    scoring: str
        Type of scoring to do the hyperparameter searching (such as 'auc_roc', 'recall',...)

    Note
    ----
    This model requires binary labels.
    """

    def __init__(self, cv=5, search=None, scoring='roc_auc'):
        self.model = KNeighborsClassifier(n_neighbors=5)
        self.cv = cv
        self.search = search
        self.scoring = scoring

    def fit(self, x, y, validation_split=0, cross_validation=None):
        model_search = self.model
        metrics = {}
        k_range = list(range(1, 31))
        weight_options = ['uniform', 'distance']

        if self.search == "grid_search":
            knn_grid = {'n_neighbors': k_range, 'weights': weight_options}
            model_search = GridSearchCV(estimator=model_search, param_grid=knn_grid,
                                        scoring=self.scoring, refit=True, cv=self.cv, verbose=3)
        elif self.search == "random_search":
            knn_random_grid = {'n_neighbors': k_range, 'weights': weight_options}
            model_search = RandomizedSearchCV(estimator=model_search, param_distributions=knn_random_grid,
                                              scoring=self.scoring, cv=self.cv, refit=True,
                                              random_state=314, verbose=3)
        else:
            pass
        self.model = model_search.fit(x, y)
        if self.search in ["grid_search", "random_search"]:
            self.model = model_search.best_estimator_
            metrics[self.scoring] = model_search.best_score_
        return metrics
