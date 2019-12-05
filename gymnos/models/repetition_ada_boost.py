#
#
#   Repetition AdaBoost
#
#

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from .mixins import SklearnMixin
from .model import Model


class RepetitionAdaBoost(SklearnMixin, Model):
    """
    AdaBoost supervised model.

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

    def __init__(self, cv=5, search=None, scoring='auc_roc'):
        self.model = AdaBoostClassifier()
        self.cv = cv
        self.search = search
        self.scoring = scoring

    def fit(self, x, y, validation_split=0, cross_validation=None):
        model_search = self.model
        metrics = {}

        if self.search == "grid_search":
            ada_boost_grid = {'n_estimators': [500, 1000, 2000], 'learning_rate': [.001, 0.01, .1]}
            model_search = GridSearchCV(estimator=model_search, param_grid=ada_boost_grid,
                                        scoring=self.scoring, refit=True, cv=self.cv, verbose=3)
        elif self.search == "random_search":
            ada_boost_random_grid = {'n_estimators': [500, 1000, 2000], 'learning_rate': [.001, 0.01, .1]}
            model_search = RandomizedSearchCV(estimator=model_search, param_distributions=ada_boost_random_grid,
                                              scoring=self.scoring, cv=self.cv, refit=True,
                                              random_state=314, verbose=3)
        else:
            pass
        self.model = model_search.fit(x, y)
        if self.search in ["grid_search", "random_search"]:
            self.model = model_search.best_estimator_
        metrics[self.scoring] = model_search.best_score_
        return metrics
