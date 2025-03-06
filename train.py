import functools
import numpy as np
import itertools
import os
from typing import Mapping
import random
import pandas as pd
import math
import matplotlib.pyplot as plt
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # For TF2.16+.

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn import runner
from tensorflow_gnn.experimental import sampler
from tensorflow_gnn.models import mt_albis
tf.get_logger().setLevel('ERROR')
print(f"Running TF-GNN {tfgnn.__version__} under TensorFlow {tf.__version__}.")

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

graph_schema = tfgnn.create_schema_pb_from_graph_spec(graph_tensor_spec)

contest_id_list = [i for i in range(1699,2064)]
contest_data = {}
user_list = {}      # handle -> User
problem_list = {}   # prob_name -> Problem
handle_list = []    # user_num -> handle
prob_name_list = [] # prob_num -> prob_name
ac_user_list = []   # source(num)
ac_prob_list = []   # target(num)
for contest_id in contest_id_list:
    file_path = f'./Datas/{contest_id}.csv'
    if not (os.path.exists(file_path)):
        continue
    contest_data[contest_id] = pd.read_csv(file_path)
    #print(f"loading contest {contest_id}")
    
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
        
max_rating_diff = 300
min_rating_diff = -300
epsilon = 1e-8
max_rating = 4000
min_rating = 400
min_contest_num = 30

contest = {}

for contest_id in contest_data:
    participants = set()
    for i in range(len(contest_data[contest_id])):
        handle = str(contest_data[contest_id]['User'][i])
        rating = int(contest_data[contest_id]['Rating'][i])
        
        if handle not in participants:
            participants.add(handle)
            if handle not in contest:
                contest[handle] = 1
            else:
                contest[handle] += 1

for contest_id in contest_data:
    print(f'processing contest {contest_id}')
    for i in range(len(contest_data[contest_id])):
        prob = str(contest_data[contest_id]['Problem Name'][i])
        difficulty = int(contest_data[contest_id]['Difficulty'][i])
        handle = str(contest_data[contest_id]['User'][i])
        rating = int(contest_data[contest_id]['Rating'][i])
        if rating-difficulty>max_rating_diff or rating-difficulty<min_rating_diff or contest[handle]<min_contest_num:
            continue
        if rating<min_rating or rating>max_rating:
            continue
        rating = (math.log(rating+epsilon)-math.log(min_rating+epsilon))/(math.log(max_rating+epsilon)-math.log(min_rating+epsilon))
        difficulty = (math.log(difficulty+epsilon)-math.log(min_rating+epsilon))/(math.log(max_rating+epsilon)-math.log(min_rating+epsilon))

        if handle not in user_list:
            user_list[handle] = User(handle,rating)
            handle_list.append(handle)
            
        if prob not in problem_list:
            problem_list[prob] = Problem(prob,difficulty)
            prob_name_list.append(prob)

        ac_user_list.append(user_list[handle].num)
        ac_prob_list.append(problem_list[prob].num)

user_num = len(user_list)
print(f'user_num = {user_num}')
problem_num = len(problem_list)
print(f'prob_num = {problem_num}')
edge_num = len(ac_user_list)
print(f'edge_num = {edge_num}')
handles = handle_list
names = prob_name_list
ratings = [user_list[handle].rating for handle in handles]
difficulties = [problem_list[prob].difficulty for prob in names]
passed_sources = ac_user_list
passed_targets = ac_prob_list
be_solved_sources = passed_targets
be_solved_targets = passed_sources
            
graph_tensor = tfgnn.GraphTensor.from_pieces(
   node_sets={
       'user': tfgnn.NodeSet.from_fields(
           sizes=tf.constant([user_num]),
           features={
               "rating": tf.constant([[tmp] for tmp in ratings], dtype=tf.float32),
           }),
       "problem": tfgnn.NodeSet.from_fields(
           sizes=tf.constant([problem_num]),
           features={
               "difficulty": tf.constant([[tmp] for tmp in difficulties], dtype=tf.float32),
           })},
   edge_sets={
       "passed": tfgnn.EdgeSet.from_fields(
           sizes=tf.constant([edge_num]),
           adjacency=tfgnn.Adjacency.from_indices(
               source=("user", tf.constant(passed_sources,dtype=tf.int64)),
               target=("problem", tf.constant(passed_targets,dtype=tf.int64)))),
       "be_solved":tfgnn.EdgeSet.from_fields(
           sizes=tf.constant([edge_num]),
           adjacency=tfgnn.Adjacency.from_indices(
               source=("problem", tf.constant(be_solved_sources,dtype=tf.int64)),
               target=("user", tf.constant(be_solved_targets,dtype=tf.int64)))),
       })

train_sampling_sizes = {
    "passed": 32,
    "be_solved":128
}
validation_sample_sizes = train_sampling_sizes.copy()

def create_sampling_model(
    full_graph_tensor: tfgnn.GraphTensor, sizes: Mapping[str, int]
) -> tf.keras.Model:

    def edge_sampler(sampling_op):
        edge_set_name = sampling_op.edge_set_name
        sample_size = sizes[edge_set_name]
        return sampler.InMemUniformEdgesSampler.from_graph_tensor(
            full_graph_tensor, edge_set_name, sample_size=sample_size
        )

    def get_features(node_set_name: tfgnn.NodeSetName):
        return sampler.InMemIndexToFeaturesAccessor.from_graph_tensor(
            full_graph_tensor, node_set_name
        )

    sampling_spec_builder = tfgnn.sampler.SamplingSpecBuilder(graph_schema)
    seed = sampling_spec_builder.seed("user")
    problem_passed_seed = seed.sample(sizes["passed"], "passed")
    user_from_problem = problem_passed_seed.sample(sizes["be_solved"], "be_solved")
    sampling_spec = sampling_spec_builder.build()

    model = sampler.create_sampling_model_from_spec(
        graph_schema, sampling_spec, edge_sampler, get_features,
        seed_node_dtype=tf.int64)

    return model

sampling_model = create_sampling_model(graph_tensor, train_sampling_sizes)

def seed_dataset(ratings_: tf.Tensor, split_name: str) -> tf.data.Dataset:
    mod_ratings = tf.constant([int(10000*tmp.numpy())%10 for tmp in ratings_], dtype=tf.int64)
    if split_name == "train":
        mask =  mod_ratings >=2
    elif split_name == "validation":
        mask = mod_ratings == 0
    elif split_name == "test":
        mask = mod_ratings == 1
    else:
        raise ValueError(f"Unknown split_name: '{split_name}'")
    seed_indices = tf.squeeze(tf.where(mask), axis=-1)
    return tf.data.Dataset.from_tensor_slices(seed_indices)

train_seeds = seed_dataset(tf.squeeze(graph_tensor.node_sets["user"]["rating"],axis=-1), "train")
val_seeds = seed_dataset(tf.squeeze(graph_tensor.node_sets["user"]["rating"],axis=-1), "validation")
test_seeds = seed_dataset(tf.squeeze(graph_tensor.node_sets["user"]["rating"],axis=-1), "test")

print(len(train_seeds))
print(len(val_seeds))
print(len(test_seeds))
train_seeds = train_seeds.batch(1) #how many seed
train_seeds = train_seeds.map(
    lambda s: tf.RaggedTensor.from_row_lengths(s, tf.ones_like(s))
)
val_seeds = val_seeds.batch(1) #
val_seeds = val_seeds.map(
    lambda s: tf.RaggedTensor.from_row_lengths(s, tf.ones_like(s))
)
test_seeds = test_seeds.batch(1) #
test_seeds = test_seeds.map(
    lambda s: tf.RaggedTensor.from_row_lengths(s, tf.ones_like(s))
)

def merge_to_components(graph: tfgnn.GraphTensor) -> tfgnn.GraphTensor:
    graph = graph.merge_batch_to_components()
    new_ratings = tf.concat([tf.constant([[0]], dtype=tf.float32), graph.node_sets['user']['rating'][1:]], axis=0)
    graph = graph.replace_features(
        node_sets={'user': {
            'rating': new_ratings
        }})
    return graph

train_graphs = train_seeds.map(sampling_model)
val_graphs = val_seeds.map(sampling_model)
test_graphs = test_seeds.map(sampling_model)

train_graphs = train_graphs.map(merge_to_components)
val_graphs = val_graphs.map(merge_to_components)
test_graphs = test_graphs.map(merge_to_components)

def gen_labels(seeds):
    labels = [[-10000]]
    cnt = 0
    for seed in seeds:
        num = seed[0][0].numpy()
        new_label = [ratings[num]]
        if labels[0][0] == -10000:
            labels[0] = new_label
        else:
            labels.append(new_label)
    ret = tf.constant(labels)
    return ret

train_labels = gen_labels(train_seeds)
val_labels = gen_labels(val_seeds)
test_labels = gen_labels(test_seeds)

def _build_model(
    graph_tensor_spec,
    node_dim=16,
    edge_dim=16,
    message_dim=64,
    next_state_dim=64,
    num_message_passing=2,
    l2_regularization=5e-4,
    dropout_rate=0.1,
):
    input_graph = tf.keras.layers.Input(type_spec=graph_tensor_spec)
    graph = input_graph.merge_batch_to_components()

    def set_initial_node_state(node_set, node_set_name):
        if node_set_name == "user":
            rating = node_set["rating"]
            masked_rating = tf.keras.layers.Masking(mask_value=0.0)(rating)
            return masked_rating
        elif node_set_name == "problem":
            return node_set["difficulty"]
        else:
            raise ValueError(f"Unexpected node set name: {node_set_name}")

    def set_initial_edge_state(edge_set, *, edge_set_name):
        return {}

    graph = tfgnn.keras.layers.MapFeatures(
        node_sets_fn=set_initial_node_state,
        edge_sets_fn=set_initial_edge_state
    )(graph)

    def dense(units, activation='relu'):
        regularizer = tf.keras.regularizers.l2(l2_regularization)
        return tf.keras.Sequential([
            tf.keras.layers.Dense(
                units,
                activation=activation,
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer
            ),
            tf.keras.layers.Dropout(dropout_rate)
        ])

    for i in range(num_message_passing):
        graph = tfgnn.keras.layers.GraphUpdate(
            node_sets={
                "user": tfgnn.keras.layers.NodeSetUpdate(
                    {"be_solved": tfgnn.keras.layers.SimpleConv(
                        message_fn=dense(message_dim),
                        reduce_type="mean",
                        receiver_tag=tfgnn.TARGET
                    )},
                    tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim))
                ),
                "problem": tfgnn.keras.layers.NodeSetUpdate(
                    {"passed": tfgnn.keras.layers.SimpleConv(
                        message_fn=dense(message_dim),
                        reduce_type="mean",
                        receiver_tag=tfgnn.TARGET
                    )},
                    tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim))
                )
            }
        )(graph)

    seed_node_states = tfgnn.keras.layers.ReadoutFirstNode(node_set_name='user')(graph)
    readout = tf.keras.layers.Dense(1,activation='linear')(seed_node_states)
    return tf.keras.Model(inputs=[input_graph], outputs=[readout])

loss = tf.keras.losses.MeanAbsoluteError()
metrics = [
    tf.keras.metrics.MeanAbsoluteError(),
    tf.keras.metrics.MeanSquaredError()
]

model = _build_model(graph_tensor_spec)
model.compile(tf.keras.optimizers.Adam(learning_rate = 0.001), loss=loss, metrics=metrics)

model.summary()

train_labels_dataset = tf.data.Dataset.from_tensor_slices(train_labels)
train_dataset = tf.data.Dataset.zip((train_graphs, train_labels_dataset))
val_labels_dataset = tf.data.Dataset.from_tensor_slices(val_labels)
val_dataset = tf.data.Dataset.zip((val_graphs, val_labels_dataset))
test_labels_dataset = tf.data.Dataset.from_tensor_slices(test_labels)
test_dataset = tf.data.Dataset.zip((test_graphs, test_labels_dataset))

batch_size = 32
train_dataset = train_dataset.batch(batch_size).repeat()
val_dataset = val_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

print("training start")
history = model.fit(train_dataset, epochs=50, steps_per_epoch=64, validation_data=val_dataset)

model_name = 'test'
model.save(model_name,save_format = 'tf')

test_loss, test_mae, test_mse = model.evaluate(test_dataset)

print(f"test loss = {test_loss}")
print(f"test mae = {test_mae}")
print(f"test mse = {test_mse}")

for k, hist in history.history.items():
    plt.title(k)
    plt.plot(hist)
    plt.savefig(f'{k}{model_name}.png')
    plt.show()
