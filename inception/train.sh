#!/bin/bash

~/tensorflow/models/research/inception/bazel-bin/inception/imagenet_train --num_gpus=1 \
                      --batch_size=32 \
                      --train_dir=/tmp/imagenet_train \
		      --max_steps=500 \	
		      --data_dir=/maui/scratch/share/imagenet_tfrecord

