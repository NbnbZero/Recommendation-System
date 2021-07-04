import math
import sys
import time
import json
import xgboost as xgb
import numpy as np
from pyspark import SparkContext, SparkConf

def computePearson(bus1_scores, bus2_scores):
    co_rated_users = list(set(bus1_scores.keys()) & set(bus2_scores.keys()))
    list1 = []
    list2 = []
    for user in co_rated_users:
        list1.append(float(bus1_scores[user]))
        list2.append(float(bus2_scores[user]))
    average_1 = sum(list1) / len(list1)
    average_2 = sum(list2) / len(list2)
    numerator = 0.0
    square_sum_1 = 0.0
    square_sum_2 = 0.0
    for score1, score2 in zip(list1,list2):
        numerator += ((score1 - average_1) * (score2 - average_2))
        square_sum_1 += ((score1 - average_1) * (score1 - average_1))
        square_sum_2 += ((score2 - average_2) * (score2 - average_2))
    if square_sum_1 == 0 or square_sum_2 == 0:
        return 0
    return numerator / (math.sqrt(square_sum_1) * math.sqrt(square_sum_2))

def predictValue(test_train_score, business_pairs_dict, business_avg_score):
    business_to_predict = test_train_score[0]
    neigbors_score_list = list(test_train_score[1])
    score_weight_list = []
    for business_score in neigbors_score_list:
        key = (business_score[0], business_to_predict)
        score_weight_list.append((float(business_score[1]), business_pairs_dict.get(key, 0))) #(score, weight)
    top_50_score_list = sorted(score_weight_list, key=lambda score_weight: score_weight[1], reverse=True)[:50]
    numerator = 0.0
    denominator = 0.0
    for score_weight in top_50_score_list:
        numerator += (score_weight[0] * score_weight[1])
        denominator += abs(score_weight[1])
    if denominator == 0 or numerator == 0:
        return [business_to_predict, business_avg_score.get(business_to_predict), len(neigbors_score_list)]

    return [business_to_predict, numerator/denominator, len(neigbors_score_list)]

def computeAverage(score_list):
    sum = 0.0
    if len(score_list) == 0:
        return 0
    for pair in score_list:
        sum += float(pair[1])
    return sum/len(score_list)

def Model_Based(pre_training_data, pre_testing_data, user_feature_file, business_feature_file):
    user_to_train = set(pre_training_data.map(lambda x: x[0]).distinct().collect())
    business_to_train = set(pre_training_data.map(lambda x: x[1]).distinct().collect())

    user_feature_map = sc.textFile(user_feature_file).map(lambda line: json.loads(line)) \
        .filter(lambda x: x['user_id'] in user_to_train) \
        .map(lambda x: (x['user_id'], [x['review_count'], x['average_stars']])) \
        .collectAsMap()

    business_feature_map = sc.textFile(business_feature_file).map(lambda line: json.loads(line)) \
        .filter(lambda x: x['business_id'] in business_to_train) \
        .map(lambda x: (x['business_id'], [x['review_count'], x['stars']])) \
        .collectAsMap()

    training_x = np.array(pre_training_data.map(lambda record: np.array([user_feature_map[record[0]], business_feature_map[record[1]]]).flatten()).collect())
    training_y = np.array(pre_training_data.map(lambda record: float(record[2])).collect())
    testing_x = np.array(pre_testing_data.map(lambda record: np.array([user_feature_map.get(record[0], [0, 2.5]),business_feature_map.get(record[1], [0, 2.5])]).flatten()).collect())

    model = xgb.XGBRegressor(max_depth=5,
                             learning_rate=0.1,
                             n_estimators=100,
                             objective='reg:linear',
                             booster='gbtree',
                             gamma=0,
                             min_child_weight=1,
                             subsample=1,
                             colsample_bytree=1,
                             reg_alpha=0,
                             reg_lambda=1,
                             random_state=0)

    model.fit(training_x, training_y)
    predicted_value = model.predict(testing_x)

    final_res = []
    for pair in zip(pre_testing_data.collect(), predicted_value):
        final_res.append(((pair[0][0],pair[0][1]), pair[1]))
    return final_res

def Collaborative_Filtering(pre_training_data, pre_testing_data):
    user_idx_dict = pre_training_data.map(lambda x: x[0]).distinct().sortBy(lambda x : x).zipWithIndex().collectAsMap()
    business_idx_dict = pre_training_data.map(lambda x: x[1]).distinct().sortBy(lambda x : x).zipWithIndex().collectAsMap()
    idx_user_dict = {idx: user for user, idx in user_idx_dict.items()}
    idx_business_dict = {idx: business for business, idx in business_idx_dict.items()}

    business_user_rdd = pre_training_data.map(lambda x: (business_idx_dict[x[1]], (user_idx_dict[x[0]], x[2])))\
        .groupByKey().mapValues(list)
    user_business_score_rdd = pre_training_data.map(lambda x: (user_idx_dict[x[0]], (business_idx_dict[x[1]], x[2])))\
        .groupByKey().mapValues(list)

    business_user_score = business_user_rdd.collectAsMap()
    user_business_score = user_business_score_rdd.collectAsMap()
    business_avg_score = business_user_rdd.map(lambda record: (record[0], computeAverage(record[1]))).collectAsMap()
    user_avg_score = user_business_score_rdd.map(lambda record: (record[0], computeAverage(record[1]))).collectAsMap()

    test_user_business_rdd = pre_testing_data.map(lambda x: (user_idx_dict.get(x[0], -1), business_idx_dict.get(x[1], -1)))\
        .filter(lambda record: record[0] != -1 and record[1] != -1)

    filtered_pairs = pre_testing_data.filter(lambda pair : pair[0] not in user_idx_dict.keys() or pair[1] not in business_idx_dict.keys()).collect()

    joined_rdd = test_user_business_rdd.leftOuterJoin(user_business_score_rdd)
    candidate_pairs = joined_rdd.flatMap(lambda record: [(bus_score[0], record[1][0]) for bus_score in record[1][1]])

    business_pairs_dict = candidate_pairs\
        .filter(lambda pair : len(set(dict(business_user_score.get(pair[0])).keys()) & set(dict(business_user_score.get(pair[1])).keys())) >= 200)\
        .map(lambda pair : (pair, computePearson(dict(business_user_score.get(pair[0])), dict(business_user_score.get(pair[1])))))\
        .filter(lambda pair: pair[1]>0).map(lambda pair : {(pair[0][0], pair[0][1]): pair[1]})\
        .flatMap(lambda pair: pair.items()).collectAsMap()

    predict_res = joined_rdd.map(lambda record : (record[0], predictValue(record[1], business_pairs_dict, business_avg_score)))
    final_res = predict_res.map(lambda pair: ((idx_user_dict[pair[0]], idx_business_dict[pair[1][0]]), pair[1][1])).collect()
    predict_val_neighbor_num = predict_res.map(lambda pair : ((idx_user_dict[pair[0]], idx_business_dict[pair[1][0]]), pair[1][2])).collectAsMap()
    for pair in filtered_pairs:
        if pair[0] in user_idx_dict.keys():
            final_res.append((tuple(pair), user_avg_score[user_idx_dict[pair[0]]]))
            predict_val_neighbor_num[tuple(pair)] = len(user_business_score[user_idx_dict[pair[0]]])
        elif pair[1] in business_idx_dict.keys():
            final_res.append((tuple(pair), business_avg_score[business_idx_dict[pair[0]]]))
            predict_val_neighbor_num[tuple(pair)] = 0
        else:
            final_res.append((tuple(pair), 2.5))
            predict_val_neighbor_num[tuple(pair)] = 0
    return final_res, predict_val_neighbor_num

start_time = time.time()
config = SparkConf().setMaster("local[*]").set('spark.executor.memory','4g').set('spark.driver.memory','4g')
sc = SparkContext.getOrCreate(config)

folder_path = ''#sys.argv[1]
user_feature_file = folder_path+'user.json'
business_feature_file = folder_path+'business.json'
training_file_path = folder_path+'yelp_train.csv'
testing_file_path = 'yelp_val_in.csv'#sys.argv[2]
output_file_path = 'output4.csv'#sys.argv[3]
text_rdd = sc.textFile(training_file_path)
first_line = text_rdd.first()

pre_training_data = sc.textFile(training_file_path).filter(lambda line : line != first_line).map(lambda line : line.split(','))
pre_testing_data = sc.textFile(testing_file_path).filter(lambda line : line != first_line).map(lambda line : line.split(','))

model_based_res = Model_Based(pre_training_data,pre_testing_data,user_feature_file,business_feature_file)
cf_res, testing_X_neighbor_num = Collaborative_Filtering(pre_training_data, pre_testing_data)
max_neighbor_num = max(testing_X_neighbor_num.values())
cf_normalized = []
mb_normalized = []
for pair in cf_res:
    cf_normalized.append((pair[0], float(testing_X_neighbor_num[pair[0]] / max_neighbor_num) * pair[1]))
for pair in model_based_res:
    mb_normalized.append((pair[0], (1 - float(testing_X_neighbor_num[pair[0]] / max_neighbor_num)) * pair[1]))

combined_res = mb_normalized + cf_normalized
combined_rdd = sc.parallelize(combined_res).reduceByKey(lambda x,y : x+y)

with open(output_file_path, 'w') as output:
    output.write('user_id,business_id,prediction\n')
    for pair in combined_rdd.collect():
        line = pair[0][0] + "," + pair[0][1] + "," + str(pair[1]) + "\n"
        output.write(line)
    output.close()

end_time = time.time()
print(end_time-start_time)
