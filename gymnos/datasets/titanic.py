#
#
#   Titanic
#
#

from .dataset import SparkDataset, ClassLabel, Array

from pyspark.sql.types import IntegerType
from pyspark.ml.feature import VectorAssembler


class Titanic(SparkDataset):
    """
    Predict survival on the Titanic.
    This dataset requires to be executed in :class:`~gymnos.execution_environments.fourth_platform.FourthPlatform`.

    Characteristics
        - **Samples total**: xxx
        - **Dimensionality**: [6]
        - **Features**: [Pclass, Age, SibSp, Parch, Fare, is_female]
    """

    @property
    def features_info(self):
        return Array(shape=[6], dtype=float)

    @property
    def labels_info(self):
        return ClassLabel(names=["not_survived", "survived"])

    def load(self):
        df = self.spark.read.format("telefonica").option("dataset.id", "titanic").option("dataset.version", 1).load()

        df = df.drop("Cabin")

        df = df.na.drop(how="any")

        df = df.withColumn("is_female", (df["Sex"] == "female").cast(IntegerType()))

        assembler = VectorAssembler(
            inputCols=["Pclass", "Age", "SibSp", "Parch", "Fare", "is_female"],
            outputCol=self.features_col
        )
        df = assembler.transform(df)

        df = df.withColumnRenamed("Survived", self.labels_col)

        return df
