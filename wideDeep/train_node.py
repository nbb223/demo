import pandas as pd
import tensorflow as tf
import numpy as np
import math
import os, json, sys
#from tensorflow.python.platform import tf_logging as logging
#logging._get_logger().setLevel(logging.INFO)
import time
start = time.clock()

model_dir = 'hdfs://default/model'

tf.app.flags.DEFINE_string("config", "", "tf_config file")
FLAGS = tf.app.flags.FLAGS
if FLAGS.config == "":
    print("Usage: python3 train_node.py --config=JSON_CONFIG_FILE")
    sys.exit(-1)

config = json.load(open(FLAGS.config, 'r'))
os.environ["TF_CONFIG"] = json.dumps(config)

# ### prameters to adjust:

hidden_units = [512,512,512]
learning_rate = 0.001
batch_size=50
num_epochs=5

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
hdfs = HDFileSystem(host='172.17.0.2', port=9000) 
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

#calculate mean and std of features in iteration fashion, just drop the last chunk for simplicity
def cal_mean_std(file_name):
  with hdfs.open(file_name) as f:
    reader = pd.read_csv(f, delimiter=delim, iterator=True, chunksize=chunksize)
    mean_arr = []
    var_arr = []
    for chunk in reader:
        if (len(chunk) == chunksize):
            mean_arr.append(chunk.mean())
            var_arr.append(chunk.var())


    df = pd.concat(mean_arr, axis=1)
    df = df.swapaxes(0,1)
    mean = df.mean()

    df = pd.concat(var_arr, axis=1)
    df = df.swapaxes(0,1)
    std = df.mean().apply(math.sqrt)
    return mean, std

def genNormalizer(mean, std):
    def func(x):
            return (tf.cast(x, tf.float32)-mean)/std
    return func

means, stds = cal_mean_std(training_data_pandas)

numerical_cols = []       #numerical feature_columns
for k in features_num:
    avg = means[k]
    std = stds[k]

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
        labels = tf.stack(labels, axis=1)
        return features, labels

    dataset = tf.data.TextLineDataset(filenames)
    dataset = dataset.skip(1).batch(batch_size=batch_size)
    dataset = dataset.map(parse_one_batch)
    dataset = dataset.repeat(num_epochs)
    return dataset

estimator = tf.estimator.DNNLinearCombinedClassifier(
    model_dir=model_dir,
    linear_feature_columns=wide_cols,
    dnn_feature_columns=deep_cols,
    dnn_hidden_units=hidden_units,
    n_classes=len(label_vocabulary), label_vocabulary=label_vocabulary)

def getTestData(test_files):
    def parse_one_batch(records):
        columns = tf.decode_csv(records, default_value, field_delim=delim)
        features = dict([(k, columns[v]) for k, v in feature_cols.items()])
        return features

    dataset = tf.data.TextLineDataset(test_files)
    dataset = dataset.skip(1).batch(batch_size=batch_size)
    dataset = dataset.map(parse_one_batch)
    return dataset

train_spec = tf.estimator.TrainSpec(input_fn=lambda:getBatches([training_data_set]))
eval_spec = tf.estimator.EvalSpec(input_fn=lambda:getBatches(test_file), start_delay_secs=10)

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
end = time.clock()
print("***** training finished, time elapsed: ", end-start, " ******")
'''
results = estimator.predict(input_fn=lambda:getTestData(test_file))

y_pred = []
for i in results:
    y_pred.append(i['classes'][0].decode())

print(y_pred)
'''
