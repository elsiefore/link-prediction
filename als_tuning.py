als_ranks = [8, 12, 16]
als_errors = [0, 0, 0]
als_models = [0, 0, 0]
index = 0
min_error = float('inf')
best_rank = -1

als_evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")

for rank in als_ranks:
    als = ALS(rank=rank, maxIter=10, regParam=0.01, userCol="user", itemCol="item", ratingCol='rating',
          coldStartStrategy="drop")
    als_model = als.fit(train_df)
    als_predictions = als_model.transform(test_df)
    als_rmse = als_evaluator.evaluate(als_predictions)
    als_errors[index] = als_rmse
    als_models[index] = als_model
    print('For rank %s the RMSE is %s' % (rank, als_rmse))
    if als_rmse < min_error:
        min_error = als_rmse
        best_rank = index
    index += 1

als.setRank(als_ranks[best_rank])
als_model = als_models[best_rank]
als_model = als.fit(train_df)

# evaluate the performance of ALS
als_predictions = als_model.transform(test_df)
als_rmse = als_evaluator.evaluate(als_predictions)

# Top 10 recommendations per user
userRecs = als_model.recommendForAllUsers(10)

print('The best model was trained with rank %s' % als_ranks[best_rank])
print("Root-mean-square error of the best rank = " + str(als_rmse))
als_recs = als_model.recommendForAllUsers(10).collect()

# output als results
with open("results/als_results.txt", 'w') as f:
    f.write(str(bytes("ranks and erros {}:{}".format(str(als_ranks), str(als_errors)), encoding='utf-8')))
    f.write("\n")
    f.write(str(bytes("The best model was trained with rank {}".format(als_ranks[best_rank]), encoding='utf-8')))
    f.write("\n")
    f.write(str(bytes("Root-mean-square error of the best rank = {}".format(str(als_rmse)), encoding='utf-8')))
    f.write("\n")
    for result in als_recs:
        f.write(str(bytes("{}: {}".format(str(result.user), str(result.recommendations)), encoding='utf-8')))
        f.write("\n")
