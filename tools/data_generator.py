#!/mnt/data/anaconda3-cpu-mkl/bin/python

import sys
import pandas as pd
from multiprocessing import Process
import os
import time

start = time.time()


# In[22]:


EXT_FEATURE_NUM = 85
EXT_CATEGORY_NUM = 200
#total data size = DATASET_SIZE * MULTIPLE
DATASET_SIZE = 6500
CONCURRENCY = 90


if len(sys.argv) == 1:
    output_file = "./expand.csv"
else :
    output_file = sys.argv[1]

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
for i in range(CONCURRENCY):
    p = Process(target=gen_data_dataset, args=(i,))
    p.start()
    proc_arr.append(p)

for i in range(CONCURRENCY):
    proc_arr[i].join()


for i in range(CONCURRENCY):
    f = output_file + str(i)
    os.system('cat ' + f  + " >> " + output_file)
    os.system('rm ' + f)

end = time.time()
print("***** data generating finished, time elapsed: ", end-start, "Secs ******")
