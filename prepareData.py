import argparse
import pandas as pd
import re
import math
from pyspark import SparkConf, SparkContext, SQLContext


def count_iterable(i):
    return sum(1 for e in i)

def pruneData(rdd, min_out_degree):
    return rdd.groupByKey().filter(lambda x: count_iterable(x[1]) >= min_out_degree)    

def split_rows(l, ratio):
    size_of_row = count_iterable(l[1])
    first_rows = int(math.floor(size_of_row * ratio))
    return (l[0], list(l[1])[:first_rows], list(l[1])[first_rows:])

def trainTestSplit(rdd, ratio = 0.8):
    split_rdd = rdd.map(lambda l: split_rows(l, ratio))
    train_rdd = split_rdd.map(lambda l: (l[0], l[1]))
    test_rdd = split_rdd.map(lambda l: (l[0], l[2]))
    return (train_rdd, test_rdd)

# read raw data
def mapLineToUserPairs(line):
    pairs = re.split(' ', line)
    return (pairs[0], pairs[1])


def main(args):
	conf = SparkConf()
	sc = SparkContext(conf=conf)
	sqlContext = SQLContext(sc)

	pd.read_csv(args.input,nrows=int(args.nrows)).to_csv(
    	'temp/nrows_twitter_links.csv',header=False,index=False)

	raw = sc.textFile("temp/nrows_twitter_links.csv")

	pruned_data = pruneData(raw.filter(lambda l: l).map(lambda l: mapLineToUserPairs(l)), int(args.minOutDegrees))

	(train_rdd, test_rdd) = trainTestSplit(pruned_data, ratio=args.trainRatio)

	train_expanded = train_rdd.flatMapValues(lambda x: x).map(lambda row: (int(row[0]), int(row[1])))
	test_expanded = test_rdd.flatMapValues(lambda x: x).map(lambda row: (int(row[0]), int(row[1])))

	train_expanded.toDF().write.option("sep"," ").csv('temp/' + args.output + '/train_links')
	test_expanded.toDF().write.option("sep"," ").csv('temp/' + args.output + '/test_links')

	sc.stop()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input')
	parser.add_argument('--output')
	parser.add_argument('--nrows', default=1e6)
	parser.add_argument('--minOutDegrees', default=4)
	parser.add_argument('--trainRatio', default=0.8)
	args = parser.parse_args()

	main(args)

	# spark-submit prepareData.py --input data/twitter_links_pruned_3M.csv --output twitter_links_pruned_3M --nrows 10000000 --minOutDegrees 1