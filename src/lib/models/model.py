#
#
#   Model
#
#

import os
import joblib

from keras import models
from ..trackers import History


class Model:

    def __init__(self, input_shape, **hyperparameters):
        self.input_shape = input_shape

    def fit(self, X, y, batch_size=32, epochs=1, callbacks=None, val_data=None, verbose=1):
        raise NotImplementedError()

    def predict(self, X, batch_size=32, verbose=0):
        raise NotImplementedError()

    def evaluate(self, X, y, batch_size=32, verbose=0):
        raise NotImplementedError()

    def restore(self, file_path):
        raise NotImplementedError()

    def save(self, directory, name="model"):
        raise NotImplementedError()


class KerasModel(Model):

    def __init__(self, input_shape, sequential_or_functional_model, **hyperparameters):
        super().__init__(input_shape)

        self.model = sequential_or_functional_model

    def compile(self, loss, optimizer, metrics=None):
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def fit(self, X, y, batch_size=32, epochs=1, callbacks=None, val_data=None, verbose=1):
        history = self.model.fit(X, y, batch_size=batch_size, epochs=epochs, callbacks=callbacks,
                                 validation_data=val_data, verbose=verbose)
        return history.history

    def predict(self, X, batch_size=32, verbose=0):
        return self.model.predict(X, batch_size=batch_size, verbose=verbose)

    def evaluate(self, X, y, batch_size=32, verbose=0):
        results = self.model.evaluate(X, y, batch_size=batch_size, verbose=verbose)
        return dict(zip(self.model.metrics_names, results))

    def save(self, directory, name="model"):
        self.model.save(os.path.join(directory, name + ".h5"))

    def restore(self, file_path):
        self.model = models.load_model(file_path)


class ScikitLearnModel(Model):

    def __init__(self, input_shape, sklearn_model, **hyperparameters):
        super().__init__(input_shape)

        self.model = sklearn_model

    def fit(self, X, y, batch_size=32, epochs=1, callbacks=None, val_data=None, verbose=1):
        self.model.fit(X, y)

        history = History()

        train_metrics = self.evaluate(X, y)

        history.log_metrics(train_metrics)

        if val_data is not None:
            val_metrics = self.evaluate(val_data[0], val_data[1])
            history.log_metrics(val_metrics, prefix="val_")

        return history.metrics

    def predict(self, X, batch_size=32, verbose=0):
        return self.model.predict(X)

    def save(self, directory, name="model"):
        joblib.dump(self.model, os.path.join(directory, name + ".joblib"))

    def restore(self, file_path):
        self.model = joblib.load(file_path)
