#!/mnt/data/anaconda3-cpu-mkl/bin/python

import sys
import pandas as pd
from multiprocessing import Process
import os
import time

start = time.time()


# In[22]:


#DATASET_SIZE = 17618
DATASET_SIZE = 1000
#CONCURRENCY = 50 
CONCURRENCY = 1

if len(sys.argv) == 1:
    output_file = "./sparse.csv"
else:
    output_file = sys.argv[1]

features = [
    'w1',
    'd0',
    'd1',
    'd2',
    'd3',
    'd4',
    'label'
]

import random

NUM_DEEP_FEATURES = 5
DEEP_CATEGORIES = 3000 #category num of deep each feature
DEEP_NON_ZERO = 10

dataset = pd.DataFrame(columns=features)
def gen_data_dataset(seq_num):
    for i in range(DATASET_SIZE):
        data_row = []

        idx = random.randint(0,99)
        '''
        w1 = [.0] * 100
        w1[idx] = random.random()
        data_row.append(w1)
        '''
        data_row.append("10:1.5")
    
        for k in range(NUM_DEEP_FEATURES):
            deep_feature = ''
            for i in range(DEEP_NON_ZERO):
                if (i < DEEP_NON_ZERO-1):
                    deep_feature += str(random.randint(0, DEEP_CATEGORIES)) + ":"
            else:
                deep_feature += str(random.randint(0, DEEP_CATEGORIES))
            data_row.append(deep_feature)
        
        data_row.append(random.randint(0,1))
        dataset.loc[idx] = data_row

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
