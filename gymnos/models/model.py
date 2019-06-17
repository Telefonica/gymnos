#
#
#   Model
#
#

import numpy as np

from collections import defaultdict
from abc import ABCMeta, abstractmethod


class Model(metaclass=ABCMeta):
    """
    Base class for all Gymnos models.

    You need to implement the following methods: ``fit``, ``predict``, ``evaluate``, ``save`` and
    ``restore``.
    """

    @abstractmethod
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

        Returns
        ------
        metrics: dict
            Training metrics
        """

    def fit_generator(self, generator, **parameters):
        """
        Fit model to training generator

        Parameters
        ----------
        generator: generator
            Generator yielding (X, y) tuples
        **parameters: any, optional
            Any parameter needed to train the model

        Returns
        -------
        metrics: dict
            Training metrics
        """
        raise NotImplementedError("Model {} don't implement fit_generator method".format(self.__class__.__name__))

    @abstractmethod
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

    @abstractmethod
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

    def evaluate_generator(self, generator):
        """
        Evaluate model performance with generator.

        Parameters
        -----------
        generator: generator
            Generator yielding features, labels

        Returns
        -------
        metrics: dict
            Dictionnary with metrics
        """
        metrics = defaultdict(list)

        for X, y in generator:
            batch_metrics = self.evaluate(X, y)
            for metric_name, metric_value in batch_metrics.items():
                metrics[metric_name].append(metric_value)

        for metric_name, metric_value in metrics.items():
            metrics[metric_name] = np.mean(metric_value)

        return metrics

    @abstractmethod
    def save(self, save_dir):
        """
        Save model to ``save_dir``.

        Parameters
        ----------
        save_dir: str
            Path (Directory) to save model.
        """

    @abstractmethod
    def restore(self, save_dir):
        """
        Restore model from ``save_dir``.

        Parameters
        ----------
        save_dir: str
            Path (Directory) where the model is saved.
        """
