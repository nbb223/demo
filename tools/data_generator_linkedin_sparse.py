import sys
import pandas as pd
from multiprocessing import Process
import os
import time
import random

start = time.time()

#DATASET_SIZE = 17618
DATASET_SIZE = 500
CONCURRENCY = 20

NUM_DEEP_FEATURES = 5
DEEP_CATEGORIES = 3000 #category num of each deep feature (in form of vectoer)
WIDE_CATEGORIES = 100  #category num of each wide feature (in form of vectoer)
DEEP_NON_ZERO = 10     #num of nono-zero vector in one deep feature


if len(sys.argv) == 1:
    output_file = "/memverge/home/songjue/data/tmp/sparse.csv"
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
deep_range = range(0, DEEP_CATEGORIES)
wide_range = range(0, WIDE_CATEGORIES)

dataset = pd.DataFrame(columns=features)
def gen_data_dataset(seq_num):
    global dataset
    for idx in range(DATASET_SIZE):
        data_row = []

        data_row.append(str(random.sample(wide_range, 1)[0]) + ":" + str(round(random.random(), 3)))

    
        for k in range(NUM_DEEP_FEATURES):
            deep_feature = ':'.join(str(e) for e in random.sample(deep_range, DEEP_NON_ZERO))
            data_row.append(deep_feature)
        
        data_row.append(random.randint(0,1))
        dataset.loc[idx] = data_row

    global output_file
    output_file = output_file + str(seq_num)
    header = False
    if (seq_num == 0):
        header = True
    return dataset.to_csv(output_file, index=False, header=header)

#gen_data_dataset(0)
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
