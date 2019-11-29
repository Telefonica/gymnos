#
#
#   Repetition Random Forest
#
#

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from .mixins import SklearnMixin
from .model import Model
from .utils.repetition_grids import RANDOM_FOREST_GRID, RANDOM_FOREST_RANDOM_GRID


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

    def __init__(self, cv, search):
        self.model = RandomForestClassifier(n_estimators=500)
        self.cv = cv
        self.search = search

    def fit(self, X, y):
        model_search = self.model
        if self.search == "grid_search":
            model_search = GridSearchCV(self.model, RANDOM_FOREST_GRID, scoring='roc_auc', cv=self.cv, n_jobs=-1,
                                        verbose=3)
        elif self.search == "random_search":
            model_search = RandomizedSearchCV(estimator=self.model, scoring='roc_auc',
                                              param_distributions=RANDOM_FOREST_RANDOM_GRID, n_iter=100, cv=3,
                                              verbose=3,
                                              random_state=42, n_jobs=-1)
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
