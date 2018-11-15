#!/bin/bash

cd /memverge/home/songjue/tensorflow/models/official/resnet

CUDA_VISIBLE_DEVICES=1 ~songjue/anaconda3-gpu/bin/python imagenet_main.py \
		--resnet_size=50 \
		--data_dir=/home/yli/nvme_ssd/imagenet/tfrecord \
		--model_dir=/tmp/resNet-model-dir \
		--batch_size=32 \
                --max_train_steps=500 \
                --clean=True \
                --train_epochs=1 
