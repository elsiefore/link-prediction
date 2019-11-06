from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import StructType, StructField, IntegerType
from pyspark.sql.functions import lit
from pyspark.ml.evaluation import RegressionEvaluator

import re
import math



def mapLineToUserPairs(line):
    pairs = re.split(' ', line)
    return (int(pairs[0]), int(pairs[1]))


def getDataframeForALS(rdd, sc):
    schema = StructType([
        StructField("user", IntegerType(), True),
        StructField("item", IntegerType(), True)])
    df = sc.createDataFrame(rdd, schema)
    df = df.withColumn("rating", lit(1))
    return df

### Top 10 recommendations per user
def convertRecResult(row):
    user = row.user
    items = []
    for item in row.recommendations:
        items.append(item.item)
    return (user, items)


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



def comparePred(row, k):
  id = row[0]
  pred = row[1][0]
  actual = row[1][1]
  counter = 0
  for i in pred:
      if i in actual:
          counter = counter + 1
          continue
  return (id, counter)


################ Start of program ################  

# initialize spark
conf = SparkConf()
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

train_path = "temp/pruned_out/train_links"
test_path = "temp/pruned_out/test_links"

train_rdd = sc.textFile(train_path)
test_rdd = sc.textFile(test_path)

train_rdd_proc = train_rdd.map(lambda row: (int(row[0]), int(row[1])))
test_rdd_proc = test_rdd.map(lambda row: (int(row[0]), int(row[1])))

train_df = getDataframeForALS(train_rdd_proc, sqlContext)
test_df = getDataframeForALS(test_rdd_proc, sqlContext)

k = 10
als = ALS(rank=100, maxIter=10, regParam=0.01, userCol="user", itemCol="item", ratingCol='rating',
          coldStartStrategy="drop")
als_model = als.fit(train_df)
user_top10_rdd = als_model.recommendForAllUsers(k).rdd.map(lambda l: convertRecResult(l))
validation_rdd = user_top10_rdd.join(test_rdd_proc.groupByKey().mapValues(list))

accuracy_result = validation_rdd.map(lambda row: comparePred(row, k))
print("Matched Predictions:\t{}\nk:\t{}".format(accuracy_result, k))
print("Predicted Nodes:", test_rdd_proc.map(lambda row: row[0]).distinct().count())

sc.stop()
