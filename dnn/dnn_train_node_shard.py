import pandas as pd
import tensorflow as tf
import numpy as np
import math
import time
import os, json, sys

#start = time.time()

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


hidden_units = [128,64]
learning_rate = 0.001
batch_size=5000
num_epochs=1
l1_regularization_strength = 0.001

training_data_pandas = '~/data/ai_data/kaggle-creditcard/creditcard.csv'
training_data_set = 'hdfs://default/data/creditcard.csv'
test_file = 'hdfs://default/data/creditcard.csv'

#model_dir = '/home/songjue/temp/adv'
delim = ','

chunksize = 100000
buffer_size = 10000

# ### some hard code:

label_vocabulary = [0, 1]
target = 'Class'
default_value = [[0.]] * 30 + [[0]] 


# ### get feature/label column names
df = pd.read_csv(training_data_pandas,  delimiter=delim, skipinitialspace=True, nrows=0)

feature_cols= {}
idx = 0
for col in df.columns:
    feature_cols[col] = idx
    idx += 1

label_cols = {target: feature_cols.pop(target)}

features_num = []
for i in df.columns:
    if i != target:
        features_num.append(i)

print("Numerical: ", features_num)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

reader = pd.read_csv(training_data_pandas, delimiter=delim, iterator=True, chunksize=chunksize)
for chunk in reader:
    scaler.partial_fit(chunk.iloc[:, :-1])

def genNormalizer(mean, std):
    def func(x):
            return (tf.cast(x, tf.float32)-mean)/std
    return func

means, stds = scaler.scale_, scaler.scale_

numerical_cols = []       #numerical feature_columns
for k in features_num:
    avg = means[feature_cols[k]]
    std = stds[feature_cols[k]]

    if (std == 0.):
        print("dropping column: ", k)
    else:
        numerical_cols.append(tf.feature_column.numeric_column(key=str(k), 
                                                            normalizer_fn=genNormalizer(avg, std)))
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


estimator = tf.estimator.DNNClassifier(
		#model_dir='/tmp/tmp1mpix5xy', 
		hidden_units=hidden_units, feature_columns = numerical_cols, 
                optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.01, l1_regularization_strength=0.001)) 

start = time.time()
estimator.train(input_fn=lambda:getBatches([training_data_set]), max_steps=10000)

end = time.time()
print("***** training finished, time elapsed: ", end-start, " ******")

#metrics = estimator.evaluate(input_fn=lambda:getBatches([test_file]))
#print("metrics: ", metrics)

def getTestData(test_files): 
    def parse_one_batch(records):
        columns = tf.decode_csv(records, default_value, field_delim=delim)
        features = dict([(k, columns[v]) for k, v in feature_cols.items()])
        return features
    
#    dataset = tf.data.TextLineDataset(test_files)
#    dataset = dataset.skip(1).batch(batch_size=batch_size)
#    dataset = dataset.map(parse_one_batch)
    d = tf.data.Dataset.from_tensor_slices(test_files)
    d = d.flat_map(lambda filename: tf.data.TextLineDataset(filename, buffer_size=buffer_size).skip(1))
    d = d.apply(tf.contrib.data.map_and_batch(parse_one_batch, batch_size))
    d = d.prefetch(1)
    return d 


start = time.time()
results = estimator.predict(input_fn=lambda:getTestData([test_file]))


y_pred = []
k = 0
for i in results:
    #y_pred.append(i['classes'][0].decode())
    k += 1

print("inference finished, time elapsed: ", time.time()-start, " ******")
#print(y_pred)

