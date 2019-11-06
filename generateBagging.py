import argparse
import math

from pyspark import SparkConf, SparkContext, SQLContext





def pseudo_random_checksum(s, precision=100, bins=100):
    x = sum([ord(c) * math.sin(i + 1) for i,c in enumerate(s)]) * precision
    return int((x - math.floor(x)) * bins)


# This is for sub-paths
def getEdgeFromString(i, delimiter=' '):
    full_list = i.split(delimiter)
    out = []
    curr = full_list[0]
    for i in range(len(full_list)-1):
        next = full_list[i+1]
        out.append([curr, next])
        curr = next
    return out


def main(args):
	conf = SparkConf()
	sc = SparkContext(conf=conf)
	sqlContext = SQLContext(sc)

	raw = sc.textFile(args.input)
	bagged_rdd = raw.map(lambda x: (pseudo_random_checksum(x, precision=int(args.precision), bins=int(args.bins)), x))\
    	.map(lambda row: [row[0]] + row[1].split(' '))
	bagged_df = bagged_rdd.toDF(["bag_id", "src", "dst"])
	bagged_df.write.partitionBy("bag_id").option("sep", args.delimiter).csv('temp/' + args.output)

	sc.stop()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input')
	parser.add_argument('--output')
	parser.add_argument('--bins', default=5)
	parser.add_argument('--precision', default=100)
	parser.add_argument('--delimiter', default=' ')

	args = parser.parse_args()

	main(args)

	# spark-submit generateBagging.py --input '/Users/zianli/Downloads/train_links_3M' --output test_bagging --bins 10