#!/bin/bash

if [ $(hostname)  == 'maui' ]; then
    export CUDA_VISIBLE_DEVICES=0
elif [ $(hostname)  == 'bigisland' ]; then
    export CUDA_VISIBLE_DEVICES=1
fi

~/tensorflow/models/research/inception/bazel-bin/inception/imagenet_train --num_gpus=1 \
                      --batch_size=64 \
                      --train_dir=/tmp/imagenet_train \
		      --max_steps=500 \
		      --data_dir=/home/yli/nvme_ssd/imagenet/tfrecord

