#!/mnt/data/anaconda3-cpu-mkl/bin/python

import pandas as pd
import tensorflow as tf
import numpy as np
import math, sys

import time
from tensorflow.python.platform import tf_logging as logging

logging._get_logger().setLevel(logging.INFO)
start = time.time()

# ### prameters to adjust:

hidden_units = [128,64,32] 
learning_rate = 0.001
batch_size=9000
num_epochs=5
l1_regularization_strength = 0.001

CATEGORY_NUM = 200
NUM_PARALLEL_BATCHES = 20
hash_bucket_size = CATEGORY_NUM

#filenames = ["./ext_1.csv"]
filenames = ["dmo:///census_extended/exp1.csv", "dmo:///census_extended/exp2.csv", "dmo:///census_extended/exp3.csv", "dmo:///census_extended/exp4.csv", "dmo:///census_extended/exp5.csv"]
training_data_pandas = "sample.csv"  #to fectch feature name/dtypes and calculate mean & std for categorical columns.
target = 'income'
delim = ','
label_vocabulary = ["<=50K", ">50K"]

model_dir = 'dmo:///model_dir/model'
tf.app.flags.DEFINE_string("num_workers", "", "num of workers")
tf.app.flags.DEFINE_string("worker_idx", "", "index of worker")
FLAGS = tf.app.flags.FLAGS
#dmo_fs_lib="/opt/MemvergeDMO/lib/libmvfs_tf.so"
dmo_fs_lib="/memverge/home/songjue/tools/lib/dmo_file_system.so"


def LoadFileSystem():
    try:
        tf.load_file_system_library(dmo_fs_lib)
        return True
    except Exception as e:
        print("[ERROR]: failed to load DMO.")
        print(e)
    return False

if LoadFileSystem() == False:
    sys.exit(-1)
else :
    print("\n*******DMO enabled********\n")

df = pd.read_csv(training_data_pandas, delimiter=",", skipinitialspace=True) #, nrows=10)

feature_cols= {}
idx = 0
for col in df.columns:
    feature_cols[col] = idx
    idx += 1

label_cols = {target: feature_cols.pop(target)}

features_categorical = []
features_num = []

default_value = []

for i in df.columns:
    if i == target:
        default_value += [[""]]
        continue
    if df[i].dtype == 'O':  #object, could be str or a mix like ['1', '0', '?', 1, 0]
        df[i] = df[i].astype(str)
        features_categorical.append(i)
        default_value += [[""]]
    else:
        features_num.append(i)
        default_value += [[0.]]
        
#to make sure column name is always str
#features_categorical = list(pd.Series(features_categorical).astype(str))
#features_num = list(pd.Series(features_num).astype(str))

#print("Categorical: ", features_categorical)
#print("Numerical: ", features_num)

# calculate emb_dim
emb_dim = []
for c in features_categorical:
#    emb_dim.append(int(math.log(len(df[c].unique()), 2)))
    emb_dim.append(int(math.log(CATEGORY_NUM, 2)))
#    emb_dim.append(32)
#print(emb_dim)

def genNormalizer(a, b):
    def func(x):
            return (tf.cast(x, tf.float16)-a)/b
    return func

numerical_cols = []
for k in features_num:
    avg = df[k].mean()
    std = df[k].std()
    if (std == 0.):
        print("dropping column: ", k)
    else:
        numerical_cols.append(tf.feature_column.numeric_column(key=str(k), 
                                                    normalizer_fn=genNormalizer(avg, std)))    
base_cols = []            
for col in features_categorical:
    base_cols.append(tf.feature_column.categorical_column_with_hash_bucket(col, hash_bucket_size=hash_bucket_size))

deep_cols = []
count = 0
for col in base_cols:
    deep_cols.append(tf.feature_column.embedding_column(col, emb_dim[count]))
    count += 1

deep_cols = deep_cols + numerical_cols
#wide_cols = crossed_cols + base_cols
wide_cols = base_cols

def getBatches(filenames):
    def parse_one_batch(records):
        columns = tf.decode_csv(records, default_value, field_delim=delim)
        features = dict([(k, columns[v]) for k, v in feature_cols.items()])
        labels = [columns[v] for _, v in label_cols.items()]
        #labels = tf.stack(labels, axis=1)
        return features, labels

    d = tf.data.Dataset.from_tensor_slices(filenames)
  #  d = d.flat_map(lambda filename: tf.data.TextLineDataset(filename, buffer_size=10000).skip(1).shard(int(FLAGS.num_workers), int(FLAGS.worker_idx)))
    d = d.flat_map(lambda filename: tf.data.TextLineDataset(filename, buffer_size=10000).skip(1))

#    d = d.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=10000, count=num_epochs))
    d = d.repeat(num_epochs)
    d = d.apply(tf.contrib.data.map_and_batch(parse_one_batch, batch_size, num_parallel_batches=NUM_PARALLEL_BATCHES))
    d = d.prefetch(1000)
    return d

config = tf.estimator.RunConfig()
config = config.replace(keep_checkpoint_max=5, save_checkpoints_steps=500)
#for Intel MKL tunning
session_config = tf.ConfigProto()
session_config.intra_op_parallelism_threads = 90
session_config.inter_op_parallelism_threads = 90
config = config.replace(session_config=session_config)

estimator = tf.estimator.DNNLinearCombinedClassifier(
    model_dir=model_dir,
    config=config,
    linear_feature_columns=wide_cols,
    dnn_feature_columns=deep_cols,
    dnn_hidden_units=hidden_units,
    n_classes=len(label_vocabulary), label_vocabulary=label_vocabulary)

estimator.train(input_fn=lambda:getBatches(filenames))

end = time.time()
print("***** training finished, time elapsed: ", end-start, " ******")
