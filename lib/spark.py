#
#
#   Spark Session
#
#

from pyspark.conf import SparkConf
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master("spark://spark-master:7077") \
    .appName("gymnos") \
    .config(conf=SparkConf()) \
    .getOrCreate()
