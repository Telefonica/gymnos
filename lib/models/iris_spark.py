#
#
#   Spark Example
#
#

from .model import SparkModel

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel

from ..spark import spark


class IrisSpark(SparkModel):

    def __init__(self, input_shape, **hyperparameters):
        super().__init__(input_shape)

        self.impurity = hyperparameters.get("impurity", "gini")
        self.max_depth = hyperparameters.get("max_depth", 10)
        self.max_bins = hyperparameters.get("max_bins", 32)

        self.model = RandomForestClassifier(labelCol="label", featuresCol="features",
                                            numTrees=hyperparameters.get("num_trees", 10),
                                            maxDepth=hyperparameters.get("max_depth", 10))

        self.evaluator = MulticlassClassificationEvaluator()

    def fit(self, X, y=None, batch_size=32, epochs=1, callbacks=None, val_data=None, verbose=1):
        self.model = self.model.fit(X)

        results = {
            "acc": self.evaluator.evaluate(self.predict(X), {self.evaluator.metricName: "accuracy"})
        }

        if val_data is not None:
            results["val_acc"] = self.evaluator.evaluate(self.predict(val_data[0]),
                                                         {self.evaluator.metricName: "accuracy"})

        return results

    def predict(self, X, batch_size=32, verbose=0):
        return self.model.transform(X)

    def evaluate(self, X, y=None, batch_size=32, verbose=0):
        return {
            "acc": self.evaluator.evaluate(self.predict(X), {self.evaluator.metricName: "accuracy"})
        }

    def restore(self, file_path):
        self.model = RandomForestClassificationModel.load(file_path)

    def save(self, file_path):
        self.model.save(spark, file_path)
