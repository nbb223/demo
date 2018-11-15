
# coding: utf-8

# In[1]:


import xgboost as xgb
#from xgboost import plot_importance
#from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

delim = ','
chunksize = 10000
num_epochs = 1
#'''
params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': 2,
    'gamma': 0.1,
    'max_depth': 6,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 8,
}
'''
params = {'silent':1, 'num_class': 2, 'objective':'multi:softmax', 
         'alpha': 0.0001, 'lambda': 1}

'''
import time
#start = time.time()


# In[2]:


#training_data = '/memverge/home/songjue/data/ai_data/kaggle-creditcard/creditcard.csv'
#training_data = '/memverge/home/songjue/creditcard.csv'
#training_data = '/memverge/home/songjue/creditcard10.csv'
training_data = '/mnt/pmem/kaggle-creditcard/creditcard10.csv'
#test_data_file =  '/memverge/home/songjue/creditcard_eval.csv'
test_data_file =  '/mnt/pmem/kaggle-creditcard/creditcard-eval-10.csv'


# In[3]:

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.mean_ =  [4.74880626e+04, 1.95869237e+00, 1.65130568e+00, 1.51625234e+00,
 1.41586609e+00, 1.38024431e+00, 1.33226875e+00, 1.23709143e+00,
 1.19435081e+00, 1.09863016e+00, 1.08884785e+00, 1.02071124e+00,
 9.99199635e-01, 9.95272483e-01, 9.58593928e-01, 9.15314405e-01,
 8.76251349e-01, 8.49335573e-01, 8.38174738e-01, 8.14039072e-01,
 7.70923671e-01, 7.34522725e-01, 7.25700286e-01, 6.24459199e-01,
 6.05646005e-01, 5.21277155e-01, 4.82226167e-01, 4.03631786e-01,
 3.30082685e-01, 2.50119670e+02]

scaler.scale_ = [4.74880626e+04, 1.95869237e+00, 1.65130568e+00, 1.51625234e+00,
 1.41586609e+00, 1.38024431e+00, 1.33226875e+00, 1.23709143e+00,
 1.19435081e+00, 1.09863016e+00, 1.08884785e+00, 1.02071124e+00,
 9.99199635e-01, 9.95272483e-01, 9.58593928e-01, 9.15314405e-01,
 8.76251349e-01, 8.49335573e-01, 8.38174738e-01, 8.14039072e-01,
 7.70923671e-01, 7.34522725e-01, 7.25700286e-01, 6.24459199e-01,
 6.05646005e-01, 5.21277155e-01, 4.82226167e-01, 4.03631786e-01,
 3.30082685e-01, 2.50119670e+02]

# In[4]:

'''
with open(training_data) as f:
    reader = pd.read_csv(f, delimiter=delim, iterator=True, chunksize=chunksize)
    for chunk in reader:
        scaler.partial_fit(chunk.iloc[:, :-1])

print("mean = ", scaler.mean_)
print("std = ", scaler.scale_)
'''
# In[5]:


start = time.time()


# In[6]:

print('*** start training *** ')
for i in range(num_epochs):
    model = None
    with open(training_data) as f:
        reader = pd.read_csv(f, delimiter=delim, iterator=True, chunksize=chunksize)
        for chunk in reader:
            features = scaler.transform(chunk.iloc[:, :-1])
          #  features = chunk.iloc[:, :-1]
            labels = chunk.iloc[:, -1]
            model = xgb.train(params, dtrain=xgb.DMatrix(features, labels), xgb_model=model)


# In[7]:


print("training finished, time elapsed: ", time.time() - start)


# In[8]:


#plot_importance(model)
#plt.show()


# In[9]:


test_data = pd.read_csv(test_data_file)
start = time.time()
X_test = test_data.iloc[:, :-1]
X_test = scaler.transform(X_test)

pred = model.predict(xgb.DMatrix(X_test))
print("inference finished, time elapsed: ", time.time() - start)
print("test data size:", len(test_data), "lines of examples.")


# In[10]:


import sklearn.metrics
print("accuracy: ", sklearn.metrics.accuracy_score(test_data.iloc[:, -1], pred))


# In[11]:


pred.fill(0)

print("accuracy_baseline: ", sklearn.metrics.accuracy_score(test_data.iloc[:, -1], pred))

