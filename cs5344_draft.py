

def pruneData(data: RDD, min_out_degree: int) -> RDD
	"""Return followers with at least min out degrees

	Args:
		data: RDD(follower, followee) <=> RDD(user, item)
	"""


def trainTestSplit(data: RDD, train_ratio: float) -> Array(RDD, RDD):
	"""Split data into training and test

	Split the data based on arbitrary rules. For example, split data in user
	partitions where the latter 20% go into test set. 
	Also make sure all the followers exist in training set

	"""



def getUserMatrix(data: RDD) -> RDD(user, Array(features)): 
	"""Retrieve user latent factor matrix

	For example, use ALS to generate latent matrix for followers. This matrix 
	can be retrieved using ALS.userFeatures()
	https://spark.apache.org/docs/2.1.2/api/python/pyspark.mllib.html#pyspark.mllib.recommendation.ALS


	Return:
		A paired RDD, where the first element is the user and the 
		second is an array of features corresponding to that user.
	"""

def getItemMatrix(data: RDD) -> RDD(item, Array(features)): 
	"""Retrieve item latent factor matrix using ALS.productFeatures()
	"""


def getTopKSimilarItemsApprox(userFeature: Array(features), itemFeature: RDD, k: int, epsilon: float) -> Array((user, item)):
	"""Recommend top k items for user

	This is implementation of the epsilon method as mentioned in paper

	Return:
		A list of items sorted by the predicted rating in descending order.
	"""



def getTopKSimilarItems(userFeature: Array(features), itemFeature: RDD, k: int) -> Array((user, item)):
	"""Recommend top k items for user

	Here would use the existing recommendProducts(user, num) method

	Return:
		A list of items sorted by the predicted rating in descending order.
	"""


def getBagging(data: RDD, method: str, f: float, mu: int) -> RDD(Array(user, item)):
	"""
	Decompose the link prediction problem into smaller pieces. 
	Repeat bagging for at least mu/f^2 times

	Arg:
		method: ["randomNodeBagging", "edgeBagging", "biasedEdgeBagging"]
		f: Fraction of total nodes
		mu: Expected times that a node pair appears in the ensemble components

	Return:
		A new RDD where each row is an array of (user, item) pairs for each bagging
	"""


def getAccuracy(prediction: RDD(user, array of k items), label: Array) -> float:
	"""
	Accuracy = # of correctly predicted links / number k of predicted links
	"""


