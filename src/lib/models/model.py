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

    Methods
    -------
    fit(X, y, **parameters)
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

    predict(X)
        Predict data.

        Parameters
        ----------
        X: array_like
            Features.

        Returns
        -------
        predictions: array_like
            Predictions from ``X``.

    evaluate(X, y)
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

    save(save_path)
        Save model to ``save_path``.

        Parameters
        ----------
        save_path: str
            Path to save model.

    restore(save_path)
        Restore model from ``save_path``.

        Parameters
        ----------
        save_path: str
            Path where the model is saved.
    """
