from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import StructType, StructField, IntegerType
from pyspark.sql.functions import lit
from pyspark.ml.evaluation import RegressionEvaluator

import re
import math
# import pandas as pd


# user defined functions
def count_iterable(i):
    return sum(1 for e in i)

def pruneData(rdd, min_out_degree):
    return rdd.groupByKey().filter(lambda x: count_iterable(x[1]) > min_out_degree)

def split_rows(l, ratio):
    size_of_row = count_iterable(l[1])
    first_rows = int(math.floor(size_of_row * ratio))
    return (l[0], list(l[1])[:first_rows], list(l[1])[first_rows:])

def trainTestSplit(rdd, ratio = 0.6):
    split_rdd = rdd.map(lambda l: split_rows(l, ratio))
    train_rdd = split_rdd.map(lambda l: (l[0], l[1]))
    test_rdd = split_rdd.map(lambda l: (l[0], l[2]))
    return (train_rdd, test_rdd)

def mapLineToUserPairs(line):
    pairs = re.split(' ', line)
    return (pairs[0], pairs[1])

def getDataframeForALS(rdd, sc):
    rdd_new = rdd.flatMapValues(lambda x: x).map(lambda row: [int(x) for x in row])
    schema = StructType([
        StructField("user", IntegerType(), True),
        StructField("item", IntegerType(), True)])
    df = sc.createDataFrame(rdd_new, schema)
    df = df.withColumn("rating", lit(1))
    return df


# initialize spark
conf = SparkConf()
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

# read raw data
raw = sc.textFile("data/links.csv").filter(lambda l: l).map(lambda l: mapLineToUserPairs(l))
pruned_data = pruneData(raw, 4)
(train_rdd, test_rdd) = trainTestSplit(pruned_data)
