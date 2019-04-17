#
#
#   Model
#
#

from sklearn.base import BaseEstimator


class Model(BaseEstimator):
    """
    Base model for all gymnos models, you must implement the following methods:
        - fit(**parameters) -> dict
        - predict(X) -> array
        - evaluate(X, y) -> dict
        - save(directory)
        - restore(directory)

    We use duck-typing but this class is defined with 2 purposes:
        - Improve readability of models
        - In case we implement some method for all models in the future
    To make it compatible with Sklearn helpers (Pipeline, GridSearchCV, etc ...) we inherit from
    sklearn.base.BaseEstimator.
    """
