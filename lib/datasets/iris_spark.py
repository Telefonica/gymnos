#
#
#   Spark Example
#
#

import os

from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler

from ..spark import spark
from .dataset import SparkDataset


class IrisSpark(SparkDataset):

    def __init__(self, cache=None):
        super().__init__(cache=None)

    def read(self, download_dir=None):
        df = spark.read.csv(os.path.join("/", "tmp", "data", "iris.csv"), header=True, inferSchema=True)

        indexer = StringIndexer(inputCol="species", outputCol="label")
        df = indexer.fit(df).transform(df)

        assembler = VectorAssembler(
            inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"],
            outputCol="features")
        df = assembler.transform(df)

        return df, None  # read must return X and y but we place everything into X
