#
#
#   Mixins
#
#

import os
import keras
import sklearn
import joblib
import tensorflow as tf

from collections.abc import Iterable
from keras.models import load_model
from sklearn.model_selection import cross_val_score, train_test_split


from ..utils.io_utils import import_from_json


KERAS_CALLBACKS_IDS_TO_MODULES_PATH = os.path.join(os.path.dirname(__file__), "..", "var", "keras",
                                                   "callbacks.json")

KERAS_MODEL_SAVE_FILENAME = "model.h5"
SKLEARN_MODEL_SAVE_FILENAME = "model.joblib"


class KerasMixin:
    """
    Mixin to write keras methods. It provides implementation for ``fit``, ``predict``, ``evaluate``,
    ``predict``, ``save`` and ``restore`` methods.
    It requires a ``self.model`` variable with your compiled Keras model.

    Attributes
    ----------
    model: keras.models.Sequential or keras.models.Model
        Compiled Keras model.
    """

    def fit(self, X, y, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, shuffle=True,
            class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None):
        """
        Parameters for Keras fit method, more info in `keras <https://keras.io/models/sequential/#fit>`_

        Parameters
        ----------
        X: array_like
            Features
        y: array_like
            Targets.
        epochs: int, optional
            Number of epochs to train.
        batch_size: int, optional
            Number of samples that will be propagated.
        verbose: int, optional
            Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        callbacks: list of dict, optional
            TODO
        validation_split: float, optional
            Fraction of the training data to be used as validation data.
        shuffle: bool, optional
            Whether to shuffle the training data before each epoch
        class_weight: dict, optional
            Dictionnary mapping class indices (integers) to a weight (float) value,
            used for weighting the loss function (during training only).
        sample_weight: array_like, optional
            Array of weights for the training samples, used for weighting the loss function.
        initial_epoch: int, optional
            Epoch at which start training (useful for resuming previous training run).
        steps_per_epoch: int, optional
            Total number of steps (batches of samples) before declaring one epoch finished and starting
            the next epoch.
        validation_steps: int, optional
            Total number of steps (batches of samples) to validate before stopping.
        Returns
        -------
        dict
            Training metrics
        """

        if callbacks is not None:
            callbacks = self.__instantiate_callbacks(callbacks)

        history = self.model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks,
                                 validation_split=validation_split, shuffle=shuffle, class_weight=class_weight,
                                 sample_weight=sample_weight, initial_epoch=initial_epoch,
                                 steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

        return history.history

    def __instantiate_callbacks(self, callbacks_config):
        callbacks = []
        for callback_config in callbacks_config:
            callback_type = callback_config.pop("type")
            CallbackClass = import_from_json(KERAS_CALLBACKS_IDS_TO_MODULES_PATH, callback_type)

            callback = CallbackClass(**callback_config)
            callbacks.append(callback)

        return callbacks


    def fit_generator(self, generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None,
                      class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=False,
                      initial_epoch=0):

        if callbacks is not None:
            callbacks = self.__instantiate_callbacks(callbacks)

        history = self.model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=verbose,
                                           callbacks=callbacks, class_weight=class_weight, workers=workers,
                                           use_multiprocessing=use_multiprocessing, shuffle=shuffle,
                                           initial_epoch=initial_epoch, max_queue_size=max_queue_size)

        return history.history

    def evaluate(self, X, y):
        """
        Evaluate model from compilation metrics.

        Parameters
        ----------
        X: array_like
            Features
        y: array_like
            Labels
        """
        metrics = self.model.evaluate(X, y)
        if not isinstance(metrics, Iterable):
            metrics = [metrics]
        return dict(zip(self.model.metrics_names, metrics))

    def predict(self, X):
        """
        Predict using Keras model.

        Parameters
        ----------
        X: array_like
            Features
        """
        return self.model.predict(X)

    def save(self, save_dir):
        """
        Save keras model to h5.

        Parameters
        ----------
        save_dir: str
            Path (Directory) to save model.
        """
        self.model.save(os.path.join(save_dir, KERAS_MODEL_SAVE_FILENAME))

    def restore(self, save_dir):
        """
        Load keras model

        Parameters
        ----------
        save_dir: str
            Path (Directory) where the model is saved.
        """
        self.model = load_model(os.path.join(save_dir, KERAS_MODEL_SAVE_FILENAME))


class SklearnMixin:
    """
    Mixin to write scikit-learn models. It provides implementation for ``fit``, ``predict``, ``evaluate``, ``save``
    and ``restore`` methods.
    It requires a ``self.model`` variable with your sklearn estimator.

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
            `cross_val_score <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html>`_.  # noqa: E501

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
        metrics = {}
        if cross_validation is not None:
            print("Computing cross validation score")
            cv_metrics = cross_val_score(self.model, X, y, **cross_validation)
            metrics["cv_" + self.metric_name] = cv_metrics

        val_data = []
        if validation_split and 0.0 < validation_split < 1.0:
            X, X_val, y, y_val = train_test_split(X, y, test_size=validation_split)
            val_data = [X_val, y_val]
            print("Using {} samples for train and {} samples for validation".format(len(X), len(X_val)))

        print("Fitting model with train data")
        self.model.fit(X, y)
        print("Computing metrics for train data")
        metrics[self.metric_name] = self.model.score(X, y)

        if val_data:
            print("Computing metrics for validation data")
            val_score = self.model.score(*val_data)
            metrics["val_" + self.metric_name] = val_score

        return metrics

    def fit_generator(self, generator):
        if hasattr(self.model, 'partial_fit'):
            for X, y in generator:
                self.model.partial_fit(X, y)

            return self.evaluate_generator(generator)
        else:
            return self.fit_generator(generator)


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

    def save(self, save_dir):
        """
        Save sklearn model using ``joblib``

        Parameters
        ----------
        save_dir: str
            Path (Directory) to save model.
        """
        joblib.dump(self.model, os.path.join(save_dir, SKLEARN_MODEL_SAVE_FILENAME))

    def restore(self, save_dir):
        """
        Restore sklearn model using ``joblib``

        Parameters
        ----------
        save_dir: str
            Path (Directory) to restore model.
        """
        self.model = joblib.load(os.path.join(save_dir, SKLEARN_MODEL_SAVE_FILENAME))


class TensorFlowSaverMixin:
    """
    Mixin to write TensorFlow models. It provides implementation for ``save`` and ``restore`` methods.
    It requires a ``self.sess`` variable with your TensorFlow session.

    Attributes
    ----------
    sess: tf.Session
        TensorFlow session.
    """

    def restore(self, save_path):
        """
        Restore session from checkpoint

        Parameters
        ----------
        save_path: str
            Path (Directory) where session is saved.
        """
        saver = tf.train.Saver()
        saver.restore(self.sess, os.path.join(save_path, "session.ckpt"))

    def save(self, save_path):
        """
        Save session.

        Parameters
        ----------
        save_path: str
            Path (Directory) to save session.
        """
        saver = tf.train.Saver()
        saver.save(self.sess, os.path.join(save_path, "session.ckpt"))