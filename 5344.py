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

raw = sc.textFile("data/links.csv").filter(lambda l: l).map(lambda l: mapLineToUserPairs(l))
pruned_data = pruneData(raw, 4)
(train_rdd, test_rdd) = trainTestSplit(pruned_data)


