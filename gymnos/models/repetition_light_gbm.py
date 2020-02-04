#
#
#   Repetition LightGBM
#
#

import numpy as np
import scipy.stats as stats
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ShuffleSplit

from .mixins import SklearnMixin
from .model import Model
from ..utils.lazy_imports import lazy_imports


class RepetitionLightGBM(SklearnMixin, Model):
    """
    LightGBM supervised model.

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
        self.model = lazy_imports.lightgbm.LGBMClassifier(learning_rate=0.1, n_estimators=1000, random_state=1000)
        self.cv = cv
        self.search = search
        self.scoring = scoring
        self.n_iter = n_iter

    def fit(self, x, y, validation_split=0, cross_validation=None):
        metrics = {}
        x = np.array(x)
        y = np.array(y)

        # create cross validation iterator
        cv = ShuffleSplit(n_splits=self.cv, test_size=0.2, random_state=0)

        if self.search == "grid_search":
            light_gbm_grid = {
                'max_depth': [3, 5, 7, 9],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'min_child_weight': [0.1, 1.0, 2.0],
            }
            self.model = GridSearchCV(estimator=self.model, param_grid=light_gbm_grid, scoring=self.scoring,
                                      cv=cv, refit=True, verbose=3, n_jobs=-1)

        elif self.search == "random_search":
            light_gbm_random_grid = {
                'max_depth': stats.randint(3, 13),  # integer between 3 and 12
                'subsample': stats.uniform(0.6, 1.0 - 0.6),  # value between 0.6 and 1.0
                'colsample_bytree': stats.uniform(0.6, 1.0 - 0.6),  # value between 0.6 and 1.0
                'min_child_weight': stats.uniform(0.1, 10.0 - 0.1),  # value between 0.1 and 10.0
            }
            # This parameter defines the number of HP points to be tested
            self.model = RandomizedSearchCV(estimator=self.model, param_distributions=light_gbm_random_grid,
                                            scoring=self.scoring, cv=cv, refit=True,
                                            random_state=14, verbose=3, n_jobs=-1, n_iter=self.n_iter)

        else:
            pass

        self.model.fit(x, y)

        if self.search in ["grid_search", "random_search"]:
            metrics[self.scoring] = self.model.best_score_
        return metrics
