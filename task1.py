import math
import random
import sys
import time
from itertools import combinations
from pyspark import SparkContext, SparkConf

def computeJaccard(set1, set2):
    return float(len(set1 & set2)) / float(len(set1 | set2))

def getBands(index_list, band_num):
    res= []
    r = math.ceil(len(index_list) / band_num)
    for i in range(0, len(index_list), r):
        res.append(tuple(sorted(index_list[i:i+r])))
    return res

def getHashFunction(a,b,m):
    def calculateByInput(x):
        return ((a*x + b) % 9999991) % m
    return calculateByInput

start = time.time()
sc = SparkContext.getOrCreate()

input_file_path = 'yelp_train.csv'#sys.argv[1]
output_file_path = 'output1.csv'#sys.argv[2]
text_rdd = sc.textFile(input_file_path)
first_line = text_rdd.first()

user_idx_rdd = text_rdd.filter(lambda x : x != first_line).map(lambda line: line.split(',')[0]).distinct().sortBy(lambda x : x).zipWithIndex()
user_idx_dict = user_idx_rdd.collectAsMap()
business_idx_dict = text_rdd.filter(lambda x : x != first_line).map(lambda line: line.split(',')[1]).distinct().sortBy(lambda x : x).zipWithIndex().collectAsMap()

hash_function_list = []
hf_num = 20
band_num = 10
a_list = random.sample(range(1,999999),hf_num)
b_list = random.sample(range(999999),hf_num)
for a,b in zip(a_list,b_list):
    hash_function = getHashFunction(a,b,len(user_idx_dict))
    hash_function_list.append(hash_function)

hashed_userid_rdd = user_idx_rdd.map(lambda item : (item[1] , [f(item[1]) for f in hash_function_list]))
user_business_rdd = text_rdd.filter(lambda x : x != first_line).map(lambda line: (user_idx_dict[line.split(',')[0]], business_idx_dict[line.split(',')[1]])).groupByKey().mapValues(list)
minHash_signiture = user_business_rdd.leftOuterJoin(hashed_userid_rdd)\
    .flatMap(lambda record : [(business_idx, record[1][1]) for business_idx in record[1][0]]).reduceByKey(lambda x,y : [min(i,j) for i,j in zip(x,y)])
candidate_pairs = set(minHash_signiture.flatMap(lambda bid_hashedUser: [(band, bid_hashedUser[0]) for band in getBands(bid_hashedUser[1], band_num)]).groupByKey().mapValues(list)
                      .map(lambda x : x[1]).filter(lambda candidates_list : len(candidates_list)>1)
                      .flatMap(lambda candidates_list : [pair for pair in combinations(candidates_list,2)]).collect())
business_user_dict = text_rdd.filter(lambda x : x != first_line).map(lambda line: (business_idx_dict[line.split(',')[1]], user_idx_dict[line.split(',')[0]])).groupByKey().mapValues(list).collectAsMap()
idx_business_dict = {idx: business for business, idx in business_idx_dict.items()}

similar_pairs = []
for pair in candidate_pairs:
    jaccard_sim = computeJaccard(set(business_user_dict[pair[0]]), set(business_user_dict[pair[1]]))
    if jaccard_sim >= 0.5:
        pair = sorted(pair)
        similar_pairs.append([idx_business_dict[pair[0]], idx_business_dict[pair[1]], str(jaccard_sim)])

similar_pairs = sorted(similar_pairs)
with open(output_file_path, 'w') as output:
    output.write('business_id1,business_id2,similarity\n')
    for items in similar_pairs:
        line = ""
        for item in items:
            line+= ','+item
        line+='\n'
        output.write(line[1:])
    output.close()

end = time.time()
print(end-start)