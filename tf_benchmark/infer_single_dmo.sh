#!/bin/bash

#### run as user 'dmo' ####
MODEL=resnet50

if [ $(hostname)  == 'maui' ]; then
    export CUDA_VISIBLE_DEVICES=0
elif [ $(hostname)  == 'bigisland' ]; then
    export CUDA_VISIBLE_DEVICES=1
fi

cd ~songjue/tensorflow/benchmarks/scripts/tf_cnn_benchmarks
~songjue/anaconda3-gpu/bin/python tf_cnn_benchmarks.py \
                --enable_dmo \
		--forward_only=true \
                --save_model_steps=0 \
		--num_gpus=1 \
		--batch_size=128 \
		--model=${MODEL} \
		--variable_update=parameter_server \
		--data_dir=dmo:///imagenet/tfrecord \
		--data_name=imagenet \
                --num_batches=500 
#		--train_dir=dmo:///resnet_model_dir
		#--train_dir=/tmp/reNet-model-dir
		#--train_dir=/home/yli/nvme_ssd/songjue/renet_model_dir 

