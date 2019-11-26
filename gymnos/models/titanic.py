#
#
#   Titanic
#
#

from .model import SparkModel

from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel


class Titanic(SparkModel):
    """
    Logistic regression to solve Titanic. This model can be useful as example to see how to implement a Gymnos SparkModel.

    Parameters
    ------------
    features_col: str
        Column name for features
    labels_col: str
        Column name for labels
    predictions_col: str
        Column name for predictions
    probabilities_col: str
        Column name for probabilities.
    """  # noqa: E501

    def __init__(self, features_col, labels_col, predictions_col="predictions", probabilities_col="probabilities"):
        self.log_reg = LogisticRegression(featuresCol=features_col,
                                          labelCol=labels_col,
                                          predictionCol=predictions_col,
                                          probabilityCol=probabilities_col)
        super().__init__(features_col, labels_col, predictions_col, probabilities_col)

    def fit(self, dataset):
        self.fitted_model_ = self.log_reg.fit(dataset)
        return {"accuracy": self.fitted_model_.summary.accuracy}

    def predict(self, dataset):
        return self.fitted_model_.transform(dataset)

    def evaluate(self, dataset):
        summary = self.fitted_model_.evaluate(dataset)
        return dict(accuracy=summary.accuracy)

    def save(self, save_dir):
        self.fitted_model_.save(save_dir)

    def restore(self, save_dir):
        self.fitted_model_ = LogisticRegressionModel.load(save_dir)
