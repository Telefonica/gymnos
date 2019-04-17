#
#
#   Sklearn Mixin
#
#

import os
import sklearn
import joblib

from ...utils.iterator_utils import count
from ...logger import get_logger

from sklearn.model_selection import cross_val_score, train_test_split


class SklearnMixin:

    @property
    def metric_name(self):
        if isinstance(self.model, sklearn.base.ClassifierMixin):
            return "accuracy"
        elif isinstance(self.model, sklearn.base.RegressionMixin):
            return "mse"
        else:
            return ""


    def fit(self, X, y, validation_split=0, cross_validation=None):
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
            logger.info("Using {} samples for train and {} samples for validation".format(count(X), count(X_val)))

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
        return self.model.predict(X)

    def evaluate(self, X, y):
        return {self.metric_name: self.model.score(X, y)}

    def save(self, directory):
        joblib.dump(self.model, os.path.join(directory, "model.joblib"))

    def restore(self, directory):
        self.model = joblib.load(os.path.join(directory, "model.joblib"))
