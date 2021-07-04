import json
import xgboost as xgb
import sys
import time
import numpy as np
from pyspark import SparkContext, SparkConf

start_time = time.time()
config = SparkConf().setMaster("local[*]").set('spark.executor.memory','4g').set('spark.driver.memory','4g')
sc = SparkContext.getOrCreate(config)

folder_path = ''#sys.argv[1]+'/'
user_feature_file = folder_path+'user.json'
business_feature_file = folder_path+'business.json'
training_file_path = folder_path+'yelp_train.csv'
testing_file_path = 'yelp_val.csv'#sys.argv[2]
output_file_path = 'output3.csv'#sys.argv[3]

first_line = sc.textFile(training_file_path).first()
pre_training_data = sc.textFile(training_file_path).filter(lambda line : line != first_line).map(lambda line : line.split(','))

user_to_train = set(pre_training_data.map(lambda x : x[0]).distinct().collect())
business_to_train  = set(pre_training_data.map(lambda x : x[1]).distinct().collect())

user_feature_map = sc.textFile(user_feature_file).map(lambda line : json.loads(line))\
    .filter(lambda x : x['user_id'] in user_to_train)\
    .map(lambda x : (x['user_id'], [x['review_count'], x['average_stars']]))\
    .collectAsMap()

business_feature_map = sc.textFile(business_feature_file).map(lambda line : json.loads(line))\
    .filter(lambda x : x['business_id'] in business_to_train)\
    .map(lambda x : (x['business_id'],[x['review_count'], x['stars']]))\
    .collectAsMap()

training_x = np.array(pre_training_data.map(lambda record : np.array([user_feature_map[record[0]], business_feature_map[record[1]]]).flatten()).collect())
training_y = np.array(pre_training_data.map(lambda record : float(record[2])).collect())
testing_x = np.array(sc.textFile(testing_file_path).filter(lambda line : line != first_line).map(lambda line : line.split(','))\
    .map(lambda record : np.array([user_feature_map.get(record[0],[0,2.5]), business_feature_map.get(record[1],[0,2.5])]).flatten()).collect())
testing_rdd = sc.textFile(testing_file_path).filter(lambda line : line != first_line).map(lambda line : line.split(','))

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

final_res_rdd = testing_rdd.map(lambda x: (x[0], x[1]))

with open(output_file_path, 'w') as output:
    output.write('user_id,business_id,prediction\n')
    for pair in zip(final_res_rdd.collect(),predicted_value):
        line = pair[0][0]+","+pair[0][1]+","+str(pair[1])+"\n"
        output.write(line)
    output.close()

end_time = time.time()
print(end_time-start_time)