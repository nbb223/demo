
# coding: utf-8

# In[ ]:


import pandas as pd
import tensorflow as tf
import numpy as np
import math
import time

#start = time.time()


# ### prameters to adjust:

# In[ ]:


hidden_units = [128,64]
learning_rate = 0.001
batch_size=5000
num_epochs=1
l1_regularization_strength = 0.001
hash_bucket_size = 1000

training_data_set = '/memverge/home/songjue/data/ai_data/kaggle-creditcard/creditcard.csv'
#test_file = '/Docker_vol/ai_data/kaggle-creditcard/creditcard_fraud_test.csv'
test_file = '/memverge/home/songjue/data/ai_data/kaggle-creditcard/creditcard.csv'

#model_dir = '/home/songjue/temp/adv'
delim = ','

chunksize = 2000


# ### some hard code:

# In[ ]:


label_vocabulary = [0, 1]
target = 'Class'
#cols_categorical = ['0', '1', '2', '3']
default_value = [[0.]] * 30 + [[0]] 


# ### get feature/label column names

# In[ ]:


df = pd.read_csv(training_data_set,  delimiter=delim, skipinitialspace=True, nrows=0)

feature_cols= {}
idx = 0
for col in df.columns:
    feature_cols[col] = idx
    idx += 1

label_cols = {target: feature_cols.pop(target)}


# In[ ]:


features_num = []

for i in df.columns:
    if i != target:
        features_num.append(i)

print("Numerical: ", features_num)


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

reader = pd.read_csv('/memverge/home/songjue/data/ai_data/kaggle-creditcard/creditcard_eval.csv', delimiter=delim, iterator=True, chunksize=chunksize)
#reader = pd.read_csv(training_data_set, delimiter=delim, iterator=True, chunksize=chunksize)
for chunk in reader:
    scaler.partial_fit(chunk.iloc[:, :-1])


# In[ ]:


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


# In[ ]:


buffer_size = 10000

def getBatches(filenames):
    def parse_one_batch(records):
        columns = tf.decode_csv(records, default_value, field_delim=delim)
        features = dict([(k, columns[v]) for k, v in feature_cols.items()])
        labels = [columns[v] for _, v in label_cols.items()]
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

# In[ ]:

start = time.time()
estimator.train(input_fn=lambda:getBatches([training_data_set]), max_steps=10000)


# In[ ]:


end = time.time()
print("***** training finished, time elapsed: ", end-start, " ******")

# In[ ]:


#metrics = estimator.evaluate(input_fn=lambda:getBatches([test_file]))
#print("metrics: ", metrics)


# In[ ]:


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

#    d = d.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=10000, count=num_epochs))
    d = d.apply(tf.contrib.data.map_and_batch(parse_one_batch, batch_size))
    d = d.prefetch(1)

    return d 


# In[ ]:


start = time.time()
results = estimator.predict(input_fn=lambda:getTestData([test_file]))


# In[ ]:


y_pred = []
k = 0
for i in results:
    #y_pred.append(i['classes'][0].decode())
    k += 1
print("inference finished, time elapsed: ", time.time()-start, " ******")


# In[ ]:


#print(y_pred)

