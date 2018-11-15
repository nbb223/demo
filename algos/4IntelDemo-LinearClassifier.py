
# coding: utf-8

# In[33]:


import pandas as pd
import tensorflow as tf
import numpy as np
import math
import time

#start = time.time()


# ### prameters to adjust:

# In[34]:


#hidden_units = [10,10]
learning_rate = 0.001
batch_size=50
num_epochs=5
l1_regularization_strength = 0.001
#hash_bucket_size = 1000

training_data_set = '/memverge/home/songjue/data/ai_data/kaggle-creditcard/creditcard.csv'
test_file = '/memverge/home/songjue/data/ai_data/kaggle-creditcard/creditcard.csv'

#test_file = '/Docker_vol/ai_data/kaggle-creditcard/creditcard_fraud_test.csv'
#model_dir = '/home/songjue/temp/adv'
delim = ','

chunksize = 10000


# ### some hard code:

# In[35]:


label_vocabulary = [0, 1]
target = 'Class'
#cols_categorical = ['0', '1', '2', '3']
default_value = [[0.]] * 30 + [[0]] 


# ### get feature/label column names

# In[36]:


df = pd.read_csv(training_data_set,  delimiter=delim, skipinitialspace=True, nrows=0)

feature_cols= {}
idx = 0
for col in df.columns:
    feature_cols[col] = idx
    idx += 1

label_cols = {target: feature_cols.pop(target)}


# In[37]:


features_num = []

for i in df.columns:
    if i != target:
        features_num.append(i)

print("Numerical: ", features_num)


# In[38]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

reader = pd.read_csv(training_data_set, delimiter=delim, iterator=True, chunksize=chunksize)
for chunk in reader:
    scaler.partial_fit(chunk.iloc[:, :-1])


# In[39]:


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


# In[40]:


def getBatches(filenames):
    def parse_one_batch(records):
        columns = tf.decode_csv(records, default_value, field_delim=delim)
        features = dict([(k, columns[v]) for k, v in feature_cols.items()])
        labels = [columns[v] for _, v in label_cols.items()]
        #labels = tf.stack(labels, axis=1)
        return features, labels

    dataset = tf.data.TextLineDataset(filenames)
    #dataset = dataset.skip(1).shard(int(FLAGS.num_workers), int(FLAGS.worker_idx))
    dataset = dataset.skip(1)
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=10000, count=num_epochs))
    dataset = dataset.apply(tf.contrib.data.map_and_batch(parse_one_batch, batch_size))
    dataset = dataset.prefetch(1)
    return dataset


# In[41]:


estimator = tf.estimator.LinearClassifier(feature_columns = numerical_cols, 
                       #   label_vocabulary = label_vocabulary,
                       #   optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.01, l1_regularization_strength=0.001)
                          )


# In[42]:


start = time.time()
estimator.train(input_fn=lambda:getBatches([training_data_set]), max_steps=10000)


# In[43]:


end = time.time()
print("***** training finished, time elapsed: ", end-start, " ******")


# In[44]:


metrics = estimator.evaluate(input_fn=lambda:getBatches(test_file))
print("metrics: ", metrics)


# In[45]:


def getTestData(test_files): 
    def parse_one_batch(records):
        columns = tf.decode_csv(records, default_value, field_delim=delim)
        features = dict([(k, columns[v]) for k, v in feature_cols.items()])
        return features
    
    dataset = tf.data.TextLineDataset(test_files)
    dataset = dataset.skip(1).batch(batch_size=batch_size)
    dataset = dataset.map(parse_one_batch)
    return dataset


# In[49]:


start = time.time()
results = estimator.predict(input_fn=lambda:getTestData(test_file))


# In[50]:


y_pred = []
for i in results:
    y_pred.append(i['classes'][0].decode())
    
print("inference finished, time elapsed: ", time.time()-start, " ******")


# In[48]:


#print(y_pred)

