#!/bin/bash

PRETRAINED_CKPT="~/tensorflow/pretrained_model/inception"
~/tensorflow/models/research/inception/bazel-bin/inception/imagenet_train --num_gpus=1 \
                      --batch_size=32 \
                      --train_dir=/tmp/imagenet_train \
		      --data_dir=/maui/scratch/share/imagenet_tfrecord \
		      --max_steps=500 \
		      --pretrained_model_checkpoint_path=${PRETRAINED_CKPT}

