from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import StructType, StructField, IntegerType
from pyspark.sql.functions import lit
from pyspark.ml.evaluation import RegressionEvaluator

import re
import math


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

# specify upper of k
k = 10

# initialize spark
conf = SparkConf()
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

# read raw data
raw = sc.textFile("data/10m.csv").filter(lambda l: l).map(lambda l: mapLineToUserPairs(l))
pruned_data = pruneData(raw, 4)
(train_rdd, test_rdd) = trainTestSplit(pruned_data)

train_df = getDataframeForALS(train_rdd, sqlContext)
test_df = getDataframeForALS(test_rdd, sqlContext)

als = ALS(rank=8, maxIter=10, regParam=0.01, userCol="user", itemCol="item", ratingCol='rating',
          coldStartStrategy="drop")
als_model = als.fit(train_df)

### Top 10 recommendations per user
def convertRecResult(row):
    user = row.user
    items = []
    for item in row.recommendations:
        items.append(item.item)
    return (user, items)
user_top10_rdd = als_model.recommendForAllUsers(k).rdd.map(lambda l: convertRecResult(l))
validation_rdd = user_top10_rdd.join(test_rdd)

# use test_df to calculate accuracy
def get_k_accuracy(row, k):
    result = []
    no_result = False
    try:
        predictions = row[1][0]
        actuals = row[1][1]
    except:
        no_result = True
    
    if not no_result and not isinstance(actuals, list):
        no_result = True

    for i in range(1, k+1):
        if no_result:
            result.append((i, 0))
        else:
            result.append((i, len([x for x in predictions[:i] if x in actuals])) / i)
    return result

als_recs = als_model.recommendForAllUsers(10).collect()

# output als results
with open("results/als_10m_results.txt", 'w') as f:
    for result in als_recs:
        f.write(str(bytes("{}: {}".format(str(result.user), str(result.recommendations)), encoding='utf-8')))
        f.write("\n")

accuracy_result = validation_rdd.flatMap(lambda l : get_k_accuracy(l, k)) \
    .mapValues(lambda v: (v, 1)) \
    .reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1])) \
    .mapValues(lambda v: v[0]/v[1]) \
    .collectAsMap()

print(accuracy_result)
