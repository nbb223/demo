import pandas as pd
import tensorflow as tf
import numpy as np
import math
import os, json, sys
#from tensorflow.python.platform import tf_logging as logging
#logging._get_logger().setLevel(logging.INFO)

import time
start = time.time()

model_dir = 'hdfs://default/model'

tf.app.flags.DEFINE_string("config", "", "tf_config file")
tf.app.flags.DEFINE_string("num_workers", "", "num of workers")
tf.app.flags.DEFINE_string("worker_idx", "", "index of worker")
FLAGS = tf.app.flags.FLAGS
if FLAGS.config == "":
    print("Usage: python3 train_node.py --config=JSON_CONFIG_FILE")
    sys.exit(-1)

config = json.load(open(FLAGS.config, 'r'))
os.environ["TF_CONFIG"] = json.dumps(config)

if config['task']['type'] == 'chief':
    is_chief = True
else:
    is_chief = False

# ### prameters to adjust:

hidden_units = [128,64]
learning_rate = 0.001
batch_size=50
num_epochs=1

l1_regularization_strength = 0.001
hash_bucket_size = 1000

#hdfs path
training_data_pandas = '/data/add.csv'
training_data_set = 'hdfs://default/data/add.csv'
test_file = 'hdfs://default/data/add.csv'

delim = ','

chunksize = 200


# ### some hard code:

# In[ ]:


label_vocabulary = ['ad.', 'nonad.']
target = '1558'
cols_categorical = ['0', '1', '2', '3']

default_value = [[0]] + [[""]] *4 + [[0.]] * 1554 + [['nonad.']] #cols ['0', '1', '2', '3'] treated as categorical


# ### get feature/label column names

# In[ ]:

from hdfs3 import HDFileSystem
hdfs = HDFileSystem(host='172.17.0.2', port=9000) #namenode and port
with hdfs.open(training_data_pandas) as f:
	df = pd.read_csv(f, delimiter=delim, index_col=0, skipinitialspace=True, nrows=0)

feature_cols= {}
idx = 1
for col in df.columns:
    feature_cols[col] = idx
    idx += 1

label_cols = {target: feature_cols.pop(target)}

features_categorical = []
features_num = []
for i in df.columns:
    if i == target:
        continue
    if i in cols_categorical:
        features_categorical.append(i)
    else:
        features_num.append(i)

print("Categorical: ", features_categorical)
print("Numerical: ", features_num)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

with hdfs.open(training_data_pandas) as f:
	reader = pd.read_csv(f, delimiter=delim, index_col=0, iterator=True, chunksize=chunksize)
	for chunk in reader:
    		scaler.partial_fit(chunk.iloc[:, 4:-1])

def genNormalizer(mean, std):
    def func(x):
            return (tf.cast(x, tf.float32)-mean)/std
    return func

means, stds = scaler.scale_, scaler.scale_

numerical_cols = []       #numerical feature_columns
for k in features_num:
    avg = means[feature_cols[k]-5]
    std = stds[feature_cols[k]-5]

    if (std == 0.):
        print("dropping column: ", k)
    else:
        numerical_cols.append(tf.feature_column.numeric_column(key=str(k),
                                                            normalizer_fn=genNormalizer(avg, std)))
base_cols = []
for col in features_categorical:
    base_cols.append(tf.feature_column.categorical_column_with_hash_bucket(col, hash_bucket_size=hash_bucket_size))

deep_cols = []
for col in base_cols:
    deep_cols.append(tf.feature_column.embedding_column(col, 9))

deep_cols = deep_cols + numerical_cols
wide_cols = base_cols

def getBatches(filenames):
    def parse_one_batch(records):
        columns = tf.decode_csv(records, default_value, field_delim=delim)
        features = dict([(k, columns[v]) for k, v in feature_cols.items()])
        labels = [columns[v] for _, v in label_cols.items()]
        #labels = tf.stack(labels, axis=1)
        return features, labels

    d = tf.data.Dataset.from_tensor_slices(filenames)
    d = d.flat_map(lambda filename: tf.data.TextLineDataset(filename, buffer_size=buffer_size).skip(1))

  #  d = d.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=10000, count=num_epochs))
    d = d.repeat(num_epochs)
    d = d.apply(tf.contrib.data.map_and_batch(parse_one_batch, batch_size))
    d = d.prefetch(1)
    return d

dnn_optimizer=tf.train.AdagradOptimizer(learning_rate)
opt = tf.train.SyncReplicasOptimizer(dnn_optimizer, replicas_to_aggregate=int(FLAGS.num_workers))
sync_replicas_hook = opt.make_session_run_hook(is_chief)

estimator = tf.estimator.DNNLinearCombinedClassifier(
    model_dir=model_dir,
    linear_feature_columns=wide_cols,
    dnn_feature_columns=deep_cols,
    dnn_optimizer=opt,
    dnn_hidden_units=hidden_units,
    n_classes=len(label_vocabulary), label_vocabulary=label_vocabulary)

def getTestData(test_files):
    def parse_one_batch(records):
        columns = tf.decode_csv(records, default_value, field_delim=delim)
        features = dict([(k, columns[v]) for k, v in feature_cols.items()])
        return features

    d = tf.data.Dataset.from_tensor_slices(test_files)
    d = d.flat_map(lambda filename: tf.data.TextLineDataset(filename, buffer_size=buffer_size).skip(1))
    d = d.apply(tf.contrib.data.map_and_batch(parse_one_batch, batch_size))
    d = d.prefetch(1)
    return d

train_spec = tf.estimator.TrainSpec(input_fn=lambda:getBatches([training_data_set]), hooks=[sync_replicas_hook])
eval_spec = tf.estimator.EvalSpec(input_fn=lambda:getBatches(test_file), start_delay_secs=10)

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
end = time.time()
print("***** training finished, time elapsed: ", end-start, " ******")

results = estimator.predict(input_fn=lambda:getTestData(test_file))

'''
y_pred = []
for i in results:
    y_pred.append(i['classes'][0].decode())

print(y_pred)
'''
