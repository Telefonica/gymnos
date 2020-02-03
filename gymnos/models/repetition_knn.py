#
#
#   Repetition KNN
#
#

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

from .mixins import SklearnMixin
from .model import Model


class RepetitionKNN(SklearnMixin, Model):
    """
    KNN supervised model.

    Parameters
    ----------
    cv: int
        Number of chunks in cross validation.
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
        self.model = KNeighborsClassifier(n_neighbors=5)
        self.cv = cv
        self.search = search
        self.scoring = scoring
        self.n_iter = n_iter

    def fit(self, x, y, validation_split=0, cross_validation=None):
        metrics = {}

        # create cross validation iterator
        cv = ShuffleSplit(n_splits=self.cv, test_size=0.2, random_state=0)

        k_range = list(range(1, 31))
        weight_options = ['uniform', 'distance']

        if self.search == "grid_search":
            knn_grid = {'n_neighbors': k_range, 'weights': weight_options}
            self.model = GridSearchCV(estimator=self.model, param_grid=knn_grid,
                                      scoring=self.scoring, refit=True, cv=cv, verbose=3, n_jobs=-1)
        elif self.search == "random_search":
            knn_random_grid = {'n_neighbors': k_range, 'weights': weight_options}
            self.model = RandomizedSearchCV(estimator=self.model, param_distributions=knn_random_grid,
                                            scoring=self.scoring, cv=cv, refit=True,
                                            random_state=14, verbose=3, n_jobs=-1, n_iter=self.n_iter)
        else:
            pass

        self.model.fit(x, y)

        if self.search in ["grid_search", "random_search"]:
            metrics[self.scoring] = self.model.best_score_
            metrics["best_params"] = self.model.best_params_
        return metrics
