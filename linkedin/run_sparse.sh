#!/bin/bash

if [ $1 == 'cpu' ]; then
    echo "cpu---------------"
    export CUDA_VISIBLE_DEVICES=''
else
    echo "gpu**************"
    export CUDA_VISIBLE_DEVICES=1
fi

CTL='numactl -m 0'


for i in {1..3}; do
    nohup time $CTL ~songjue/anaconda3-tf1.12-gpu/bin/python train_sparse.py &
done

CTL='numactl -m 1'

for i in {1..3}; do
    nohup time $CTL ~songjue/anaconda3-tf1.12-gpu/bin/python train_sparse.py &
done



