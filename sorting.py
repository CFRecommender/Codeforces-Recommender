import functools
from cv2 import sort
import numpy as np
import itertools
import os
import re
from typing import Mapping
import random
import pandas as pd
import math
from datetime import datetime
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # For TF2.16+.

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn import runner
from tensorflow_gnn.experimental import sampler
from tensorflow_gnn.models import mt_albis

graph_tensor_spec = tfgnn.GraphTensorSpec.from_piece_specs(
    node_sets_spec={
        'user':
            tfgnn.NodeSetSpec.from_field_specs(
                features_spec={
                    'rating': tf.TensorSpec(shape=[None,1], dtype=tf.float32),
                },
                sizes_spec=tf.TensorSpec((1,), tf.int64)),
        'problem':
            tfgnn.NodeSetSpec.from_field_specs(
                features_spec={
                    'difficulty': tf.TensorSpec(shape=[None,1], dtype=tf.float32),
                },
                sizes_spec=tf.TensorSpec((1,), tf.int64))
    },
    edge_sets_spec={
        'passed':
            tfgnn.EdgeSetSpec.from_field_specs(
                sizes_spec=tf.TensorSpec((1,), tf.int64),
                adjacency_spec=tfgnn.AdjacencySpec.from_incident_node_sets('user', 'problem')
            ),
        'be_solved':
            tfgnn.EdgeSetSpec.from_field_specs(
                sizes_spec=tf.TensorSpec((1,), tf.int64),
                adjacency_spec=tfgnn.AdjacencySpec.from_incident_node_sets('problem', 'user')
            )
    })

contest_id_list = [i for i in range(1699,2064)]
contest_data = {}
user_list = {}      # handle -> User
problem_list = {}   # prob_name -> Problem
handle_list = []    # user_num -> handle
prob_name_list = [] # prob_num -> prob_name
g_user = []
g_prob = []
contest_cnt = {}

class User:
    def __init__(self,handle,rating):
        self.handle = handle
        self.rating = rating
        self.num = len(user_list)
    

class Problem:
    def __init__(self,name,difficulty):
        self.name = name
        self.difficulty = difficulty
        self.num = len(problem_list)

for contest_id in contest_id_list:
    file_path = f'./Datas/{contest_id}.csv'
    if not (os.path.exists(file_path)):
        continue
    contest_data[contest_id] = pd.read_csv(file_path)
    print(f"loading contest {contest_id}")
    
max_rating_diff = 300
min_rating_diff = -300
epsilon = 1e-8
max_rating = 4000
min_rating = 400
passed_size = 32
be_solved_size = 128

def normalize(x):
    return (math.log(x+epsilon)-math.log(min_rating+epsilon))/(math.log(max_rating+epsilon)-math.log(min_rating+epsilon))

for contest_id in contest_data:
    print(f'processing contest {contest_id}')
    participants = set()
    for i in range(len(contest_data[contest_id])):
        handle = str(contest_data[contest_id]['User'][i])
        rating = int(contest_data[contest_id]['Rating'][i])
        
        if handle not in participants:
            participants.add(handle)
            if handle not in contest_cnt:
                contest_cnt[handle] = 1
            else:
                contest_cnt[handle] += 1

for contest_id in contest_data:
    print(f'processing contest {contest_id}')
    for i in range(len(contest_data[contest_id])):
        prob = str(contest_data[contest_id]['Problem Name'][i])
        difficulty = int(contest_data[contest_id]['Difficulty'][i])
        handle = str(contest_data[contest_id]['User'][i])
        rating = int(contest_data[contest_id]['Rating'][i])
        '''
        if rating-difficulty>max_rating_diff or rating-difficulty<min_rating_diff:
            continue
        '''
        if rating<min_rating or rating>max_rating or contest_cnt[handle]<1:
            continue
        
        rating = normalize(rating)
        difficulty = normalize(difficulty)

        if handle not in user_list:
            user_list[handle] = User(handle,rating)
            handle_list.append(handle)
            g_user.append([])
            
        if prob not in problem_list:
            problem_list[prob] = Problem(prob,difficulty)
            prob_name_list.append(prob)
            g_prob.append([])
        
        user_num = user_list[handle].num
        prob_num = problem_list[prob].num
        
        if(len(g_user[user_num])<passed_size-1):
            g_user[user_num].append(prob_num)
        if(len(g_prob[prob_num])<be_solved_size):
            g_prob[prob_num].append(user_num)
        

model = tf.keras.models.load_model('25')
def merge_to_components(graph: tfgnn.GraphTensor) -> tfgnn.GraphTensor:
    graph = graph.merge_batch_to_components()
    new_ratings = tf.concat([tf.constant([[0]], dtype=tf.float32), graph.node_sets['user']['rating'][1:]], axis=0)
    graph = graph.replace_features(
        node_sets={'user': {
        'rating': new_ratings,
    }})
    return graph

def gen_tensor(user_cnt,user_ratings,prob_cnt,prob_diff,edge_cnt,edge_source,edge_target):
    ratings_tensor = tf.convert_to_tensor(user_ratings, dtype=tf.float32)
    ratings_tensor = tf.reshape(ratings_tensor, [-1, 1])
    difficulties_tensor = tf.convert_to_tensor(prob_diff, dtype=tf.float32)
    difficulties_tensor = tf.reshape(difficulties_tensor, [-1, 1])
    tensor = tfgnn.GraphTensor.from_pieces(
        node_sets={
            'user': tfgnn.NodeSet.from_fields(
                sizes=tf.constant([user_cnt]),
                features={
                    "rating": ratings_tensor
                }),
            "problem": tfgnn.NodeSet.from_fields(
                sizes=tf.constant([prob_cnt]),
                features={
                    "difficulty": difficulties_tensor
                })},
        edge_sets={
            "passed": tfgnn.EdgeSet.from_fields(
                sizes=tf.constant([edge_cnt]),
                adjacency=tfgnn.Adjacency.from_indices(
                    source=("user", tf.constant(edge_source, dtype=tf.int64)),
                    target=("problem", tf.constant(edge_target, dtype=tf.int64)))),
            "be_solved":tfgnn.EdgeSet.from_fields(
                sizes=tf.constant([edge_cnt]),
                adjacency=tfgnn.Adjacency.from_indices(
                    source=("problem", tf.constant(edge_target, dtype=tf.int64)),
                    target=("user", tf.constant(edge_source, dtype=tf.int64)))),
            })
    return tensor

tensors = []

def graph_generator():
    for graph in tensors:  
        yield graph



def sort_pb(user,problemset):
    global tensors
    tensors = []
    user_id = user_list[user].num
    ids = {user_id}
    id_here = {0:0}
    prob_id_here = {}
    ratings = [user_list[user].rating]
    difficulties = []
    passed_source = []
    passed_target = []
    user_num = 1
    prob_num = 0
    for passed_prob in g_user[user_id]:
        prob_id_here[passed_prob] = prob_num
        new_name = prob_name_list[passed_prob]
        difficulties.append(problem_list[new_name].difficulty)
        passed_source.append(0)
        passed_target.append(prob_num)
        
        cnt = 0
        for other_user in g_prob[passed_prob]:
            if cnt>be_solved_size/len(g_user[user_id]):
                continue
            cnt += 1
            if other_user not in ids:
                id_here[other_user] = user_num
                ids.add(other_user)
                new_handle = handle_list[other_user]
                ratings.append(user_list[new_handle].rating)
                user_num += 1
            
            passed_source.append(id_here[other_user])
            passed_target.append(prob_num)
        
        prob_num += 1
    
    edge_num = len(passed_target)
    result = {}
    
    for new_problem in problemset:
        if new_problem not in problem_list:
            print(new_problem)
            continue
        new_prob_id = problem_list[new_problem].num
        new_passed_source = passed_source.copy()
        new_passed_target = passed_target.copy()
        new_ratings = ratings.copy()
        new_user_num = user_num
        new_passed_source.append(0)
        new_passed_target.append(prob_num)
        for other_user in g_prob[new_prob_id]:
            if other_user not in ids:
                new_passed_source.append(user_num)
                new_passed_target.append(prob_num)
                new_handle = handle_list[other_user]
                new_ratings.append(user_list[new_handle].rating)
                new_user_num += 1
            else:
                new_passed_source.append(id_here[other_user])
                new_passed_target.append(prob_num)
        new_edge_num = len(new_passed_source)
        new_difficulties = difficulties.copy()
        
        new_difficulties.append(problem_list[new_problem].difficulty)
        graph_tensor = gen_tensor(new_user_num,new_ratings,prob_num+1,new_difficulties,new_edge_num,new_passed_source,new_passed_target)
        tensors.append(merge_to_components(graph_tensor))

    dataset = tf.data.Dataset.from_generator(
        graph_generator,
       output_signature=graph_tensor_spec
    )
    pre = model.predict(dataset)
    for i in range(len(problemset)):
        result[problemset[i]] = pre[i][0]
    return sorted(problemset, key = lambda prob:-result[prob])


'''
start_time = datetime.now()
pbs = ['Game of Mathletes','Bugged Sort','Multiplicative Arrays','Graph Composition','Subtract Min Sort','Farmer John\'s Card Game']
#temp = sorting('Jayanth1278',)
#for pb in pbs:
    #temp = sorting('Jayanth1278',['Bugged Sort'])
    #print(temp)
temp = sorting('Jayanth1278',pbs)
#print(temp)
#temp = sorting('Jayanth1278',['Game of Mathletes' for _ in range(1000)])
print(temp)
end_time = datetime.now()
print(end_time-start_time)
'''