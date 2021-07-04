import math
import sys
import time
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
        return (business_to_predict, business_avg_score.get(business_to_predict))

    return (business_to_predict, numerator/denominator)

def computeAverage(score_list):
    sum = 0.0
    for pair in score_list:
        sum += float(pair[1])
    return sum/len(score_list)

start_time = time.time()
config = SparkConf().setMaster("local[*]").set('spark.executor.memory','4g').set('spark.driver.memory','4g')
sc = SparkContext.getOrCreate(config)

train_file_path = 'yelp_train.csv'#sys.argv[1]+'/'
test_file_path = 'yelp_val_in.csv'#sys.argv[2]
output_file_path = 'output2.csv'#sys.argv[3]
text_rdd = sc.textFile(train_file_path)
test_rdd = sc.textFile(test_file_path)
first_line = text_rdd.first()

user_idx_dict = text_rdd.filter(lambda x : x != first_line)\
    .map(lambda line: line.split(',')[0]).distinct().sortBy(lambda x : x).zipWithIndex().collectAsMap()
business_idx_dict = text_rdd.filter(lambda x : x != first_line)\
    .map(lambda line: line.split(',')[1]).distinct().sortBy(lambda x : x).zipWithIndex().collectAsMap()
idx_user_dict = {idx: user for user, idx in user_idx_dict.items()}
idx_business_dict = {idx: business for business, idx in business_idx_dict.items()}

business_user_rdd = text_rdd.filter(lambda x : x != first_line)\
    .map(lambda line: (business_idx_dict[line.split(',')[1]], (user_idx_dict[line.split(',')[0]], line.split(',')[2])))\
    .groupByKey().mapValues(list)
business_user_score = business_user_rdd.collectAsMap()

business_avg_score = business_user_rdd.map(lambda record: (record[0], computeAverage(record[1]))).collectAsMap()

test_user_business_rdd = test_rdd.filter(lambda x : x != first_line)\
    .map(lambda line: (user_idx_dict.get(line.split(',')[0], -1), business_idx_dict.get(line.split(',')[1], -1)))\
    .filter(lambda record: record[0] != -1 and record[1] != -1)

filtered_pairs = test_rdd.filter(lambda x : x != first_line)\
    .map(lambda line : line.split(','))\
    .filter(lambda pair : pair[0] not in user_idx_dict or pair[1] not in business_idx_dict).collect()

user_business_score_rdd = text_rdd.filter(lambda x : x != first_line)\
    .map(lambda line: (user_idx_dict[line.split(',')[0]], (business_idx_dict[line.split(',')[1]], line.split(',')[2])))\
    .groupByKey().mapValues(list)

user_business_score = user_business_score_rdd.collectAsMap()
user_avg_score = user_business_score_rdd.map(lambda record: (record[0], computeAverage(record[1]))).collectAsMap()

joined_rdd = test_user_business_rdd.leftOuterJoin(user_business_score_rdd)
candidate_pairs = joined_rdd.flatMap(lambda record: [(bus_score[0], record[1][0]) for bus_score in record[1][1]])

business_pairs_dict = candidate_pairs\
    .filter(lambda pair : len(set(dict(business_user_score.get(pair[0])).keys()) & set(dict(business_user_score.get(pair[1])).keys())) >= 300)\
    .map(lambda pair : (pair, computePearson(dict(business_user_score.get(pair[0])), dict(business_user_score.get(pair[1])))))\
    .filter(lambda pair: pair[1]>0).map(lambda pair : {(pair[0][0], pair[0][1]): pair[1]})\
    .flatMap(lambda pair: pair.items()).collectAsMap()

final_res_rdd = joined_rdd.map(lambda record : (record[0], predictValue(record[1], business_pairs_dict, business_avg_score)))

with open(output_file_path, 'w') as output:
    output.write('user_id,business_id,prediction\n')
    for pair in final_res_rdd.collect():
        line = idx_user_dict[pair[0]]+","+idx_business_dict[pair[1][0]]+","+str(pair[1][1])+"\n"
        output.write(line)
    for pair in filtered_pairs:
        if pair[0] in user_idx_dict.keys():
            line = pair[0]+","+pair[1]+","+str(user_avg_score[user_idx_dict[pair[0]]])+"\n"
        elif pair[1] in business_idx_dict.keys():
            line = pair[0]+","+pair[1]+","+str(business_avg_score[business_idx_dict[pair[0]]])+"\n"
        else:
            line = pair[0]+","+pair[1]+","+str(0.0)+"\n"
        output.write(line)
    output.close()

end_time = time.time()
print(end_time-start_time)
