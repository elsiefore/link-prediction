from pyspark import SparkConf, SparkContext
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

# initialize spark
conf = SparkConf()
sc = SparkContext(conf=conf)


# read raw data
def mapLineToUserPairs(line):
    pairs = re.split(' ', line)
    return (pairs[0], pairs[1])

raw = sc.textFile("links-anon.txt").filter(lambda l: l).map(lambda l: mapLineToUserPairs(l))

# generate stats
link_count = raw.map(lambda l: ("total count", 1)).reduceByKey(lambda x, y: x + y).collect()
with open("result_stats.txt", 'w') as f:
    for item in link_count:
        f.write("%s: %s" % (item[0], item[1]))

users_out_degree = raw.map(lambda l: (l[0], 1)).reduceByKey(lambda x, y: x + y).map(lambda l: (l[1], 1)).reduceByKey(lambda x, y: x + y).collect()
with open("result_out_degree.csv", 'w') as f:
    for item in users_out_degree:
        f.write("%s, %s\n" % (item[0], item[1]))
users_in_degree = raw.map(lambda l: (l[1], 1)).reduceByKey(lambda x, y: x + y).map(lambda l: (l[1], 1)).reduceByKey(lambda x, y: x + y).collect()
with open("result_in_degree.csv", 'w') as f:
    for item in users_in_degree:
        f.write("%s, %s\n" % (item[0], item[1]))
