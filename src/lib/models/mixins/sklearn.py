#
#
#   Sklearn Mixin
#
#

import os
import sklearn
import joblib

from ...logger import get_logger

from sklearn.model_selection import cross_val_score, train_test_split


class SklearnMixin:
    """
    Mixin to write scikit-learn methods. It provides implementation for ``fit``, ``predict``, ``evaluate``, ``save``
    and ``restore`` methods.

    Attributes
    ----------
    model: sklearn.BaseEstimator
        scikit-learn estimator.
    """

    @property
    def metric_name(self):
        if isinstance(self.model, sklearn.base.ClassifierMixin):
            return "accuracy"
        elif isinstance(self.model, sklearn.base.RegressorMixin):
            return "mse"
        else:
            return ""


    def fit(self, X, y, validation_split=0, cross_validation=None):
        """
        Parameters
        ----------
        X: `array_like`
            Features.
        y: `array_like`
            Targets.
        validation_split: float, optional
            Fraction of the training data to be used as validation data.
        cross_validation: dict, optional
            Whether or not compute cross validation score. If not provided, cross-validation
            score is not computed. If provided, dictionnary with parameters for
            `cross_val_score <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html>`_.

        Examples
        --------
        .. code-block:: py

            fit(X, y, cross_validation={
                "cv": 5,
                "n_jobs": -1,
                "verbose": 1
            })

        Returns
        -------
            metrics: dict
                Training metrics.
        """

        logger = get_logger(prefix=self)

        metrics = {}
        if cross_validation is not None:
            logger.info("Computing cross validation score")
            cv_metrics = cross_val_score(self.model, X, y, **cross_validation)
            metrics["cv_" + self.metric_name] = cv_metrics

        val_data = []
        if validation_split and 0.0 < validation_split < 1.0:
            X, X_val, y, y_val = train_test_split(X, y, test_size=validation_split)
            val_data = [X_val, y_val]
            logger.info("Using {} samples for train and {} samples for validation".format(len(X), len(X_val)))

        logger.info("Fitting model with train data")
        self.model.fit(X, y)
        logger.info("Computing metrics for train data")
        metrics[self.metric_name] = self.model.score(X, y)

        if val_data:
            logger.info("Computing metrics for validation data")
            val_score = self.model.score(*val_data)
            metrics["val_" + self.metric_name] = val_score

        return metrics

    def predict(self, X):
        """
        Predict data using scikit-learn model.

        Parameters
        ----------
        X: array_like
            Features

        Returns
        -------
        predictions: array_like
            Predictions from ``X``
        """
        return self.model.predict(X)

    def evaluate(self, X, y):
        """
        Evaluate data using sklearn ``score`` method.

        Parameters
        ----------
        X: array_like
            Features
        y: array_like
            True labels
        """
        return {self.metric_name: self.model.score(X, y)}

    def save(self, save_path):
        """
        Save sklearn model using ``joblib``

        Parameters
        ----------
        save_path: str
            Path (Directory) to save model.
        """
        joblib.dump(self.model, os.path.join(save_path, "model.joblib"))

    def restore(self, save_path):
        """
        Restore sklearn model using ``joblib``

        Parameters
        ----------
        save_path: str
            Path (Directory) to restore model.
        """
        self.model = joblib.load(os.path.join(save_path, "model.joblib"))
