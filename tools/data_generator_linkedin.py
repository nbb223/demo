# Generating dummy features & records per LinkedIn's request: 
# 1. extend to 100 features with 85 categorical dummy features
# 2. each categorical feature has 200 categories
# 3. expand data size to 256/512MB per file
#

import pandas as pd
from multiprocessing import Process
import os
import time

start = time.clock()


# In[22]:


EXT_FEATURE_NUM = 85
EXT_CATEGORY_NUM = 200
#total data size = DATASET_SIZE * MULTIPLE
DATASET_SIZE = 2500
MULTIPLE = 40

output_file = "./expand.csv"


# In[23]:


features = ['age',
 'workclass',
 'fnlwgt',
 'education',
 'education.num',
 'marital.status',
 'occupation',
 'relationship',
 'race',
 'sex',
 'capital.gain',
 'capital.loss',
 'hours.per.week',
 'native.country',
 'income']

for f in range(EXT_FEATURE_NUM):
    features.append("ext_" + str(f))


# In[24]:


types = ['int64',
 'object',
 'int64',
 'object',
 'int64',
 'object',
 'object',
 'object',
 'object',
 'object',
 'int64',
 'int64',
 'int64',
 'object',
 'object']

types = types + ['object' for i in range(EXT_FEATURE_NUM)]


# In[25]:


feature_dtype_dict = dict(zip(features, types))


# In[26]:


import random, string
feature_vocabulary = {}

for (f, d) in feature_dtype_dict.items():
    if d == 'object':
        feature_vocabulary[f] = [''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5)) for i in range(EXT_CATEGORY_NUM)]

feature_vocabulary['income'] = ['<=50K', '>50K']


# In[30]:


import random

num = 0

dataset = pd.DataFrame(columns=features)
def gen_data_dataset(seq_num):
    for idx in range(DATASET_SIZE):
        data_row = []
        for f in feature_dtype_dict:
            if feature_dtype_dict[f] == 'int64':
                data_row.append(random.randint(0,100))
            else:
               # data_row.append(feature_vocabulary[f][random.randint(0, EXT_CATEGORY_NUM - 1)])
                data_row.append(feature_vocabulary[f][random.randint(0, len(feature_vocabulary[f])- 1)])

        dataset.loc[idx] = data_row

        global num
        num += 1
        if (num % 1000 == 0):
            print("num => ", num)

    global output_file
    output_file = output_file + str(seq_num)
    header = False
    if (seq_num == 0):
        header = True
    return dataset.to_csv(output_file, index=False, header=header)


# In[31]:

proc_arr = []
for i in range(MULTIPLE):
    p = Process(target=gen_data_dataset, args=(i,))
    p.start()
    proc_arr.append(p)

for i in range(MULTIPLE):
    proc_arr[i].join()


for i in range(MULTIPLE):
    os.system('cat ' + output_file + str(i)  + " >> _expand.csv")

end = time.clock()
print("***** data generating finished, time elapsed: ", end-start, "Secs ******")

