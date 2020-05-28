#
#
#   Model
#
#

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

    def predict_proba(self, X):
        """
        Predict probabilities (classification tasks).

        Parameters
        ----------
        X: array_like
            Features

        Returns
        -------
        predictions: array_like
            Label probabilities from ``X``.
        """
        raise NotImplementedError("Model {} don't implement predict_proba method".format(self.__class__.__name__))

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
        raise NotImplementedError("Model {} don't implement evaluate_generator method".format(self.__class__.__name__))

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


class SparkModel(metaclass=ABCMeta):
    """
    Base class for all Gymnos Spark models.
    You need to implement the following methods: fit, predict, evaluate, save and restore

    Parameters
    ------------
    features_col: str
        Column name for your features
    labels_col: str
        Column name for your labels
    predictions_col: str
        Column name for your predictions
    probabilities_col: str
        Column name for your probabilities
    """

    def __init__(self, features_col, labels_col, predictions_col="predictions",
                 probabilities_col="probabilities"):
        self.features_col = features_col
        self.labels_col = labels_col
        self.predictions_col = predictions_col
        self.probabilities_col = probabilities_col

    @abstractmethod
    def fit(self, dataset, **kwargs):
        """
        Returns dictionnary with metrics

        Parameters
        -----------
        dataset: pyspark.sql.DataFrame
            Dataset to fit.
        """

    @abstractmethod
    def predict(self, dataset):
        """
        Returns dataframe with predictions in ``self.predictions_col`` column.

        Parameters
        -----------
        dataset: pyspark.sql.DataFrame
            Dataset to fit.
        """

    def predict_proba(self, dataset):
        """
        Returns dataframe with probabilities in ``self.probabilities_col`` column.

        Parameters
        -----------
        dataset: pyspark.sql.DataFrame
            Dataset to fit.
        """

    @abstractmethod
    def evaluate(self, dataset):
        """
        Returns dictionnary with metrics

        Parameters
        -----------
        dataset: pyspark.sql.DataFrame
            Dataset to fit.

        Returns
        ---------
        metrics: dict
            Dictionnary with metrics
        """

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
