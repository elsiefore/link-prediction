# link-prediction
Assignment Repo

*Instructions*
- Use data/twitter_links_pruned_3M.csv as the main data source. This data is from Twitter 2010, and pruned by:
	- First 3 millions edges
	- Source nodes with degrees (in and out) between 10 and 500
- Use prepareData.py to split into train and test sets
- [Optional] Use generateBagging.py to create baggings
- Use random_walk.sh to generate Node2Vec embeddings, default 128 dimensions. 
- TODO: Implement top-k prediction with tolerance