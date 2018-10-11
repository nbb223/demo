#!/bin/bash

CUDA_VISIBLE_DEVICES=1 ~songjue/tensorflow/models/research/inception/bazel-bin/inception/imagenet_train --num_gpus=1 \
                      --batch_size=32 \
                      --train_dir=hdfs://aep0:4545/inception_model \
		      ----max_steps=500 \
		      --data_dir=hdfs://aep0:4545/imageNet/tfrecord

