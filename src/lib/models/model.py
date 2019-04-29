#
#
#   Model
#
#

from sklearn.base import BaseEstimator


class Model(BaseEstimator):
    """
    Base class for all Gymnos models.

    You need to implement the following methods: ``fit``, ``predict``, ``evaluate``, ``save`` and
    ``restore``.
    """

    def fit(self, X, y, **parameters):
        """
        Fit model to training data.

        Parameters
        ----------
        X: array_like
            Features.
        y: array_like
            Labels
        **parameters: any, optional
            Any parameter needed train the model.

        Return
        ------
        metrics: dict
            Training metrics
        """
        return super().fit(X, y, **parameters)

    def predict(self, X):
        """
        Predict data.

        Parameters
        ----------
        X: array_like
            Features.

        Returns
        -------
        predictions: array_like
            Predictions from ``X``.
        """
        return super().predict(X)

    def evaluate(self, X, y):
        """
        Evaluate model performance.

        Parameters
        ----------
        X: array_like
            Features.
        y: array_like
            True labels.

        Returns
        -------
        metrics: dict
            Dictionnary with metrics.
        """
        return super().evaluate(X, y)

    def save(self, save_path):
        """
        Save model to ``save_path``.

        Parameters
        ----------
        save_path: str
            Path (Directory) to save model.
        """
        return super().save(save_path)

    def restore(self, save_path):
        """
        Restore model from ``save_path``.

        Parameters
        ----------
        save_path: str
            Path (Directory) where the model is saved.
        """
        return super().restore(save_path)
