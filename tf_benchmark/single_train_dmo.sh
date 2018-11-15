#!/bin/bash

cd ~songjue/tensorflow/benchmarks/scripts/tf_cnn_benchmarks

if [ $(hostname)  == 'maui' ]; then
    export CUDA_VISIBLE_DEVICES=0
elif [ $(hostname)  == 'bigisland' ]; then
    export CUDA_VISIBLE_DEVICES=1
fi

#must set --data_name=imagenet to run training
~songjue/anaconda3-gpu/bin/python tf_cnn_benchmarks.py \
		--enable_dmo \
		--save_model_steps=200 \
		--num_gpus=1 \
		--batch_size=128 \
		--model=resnet50 \
		--variable_update=parameter_server \
		--save_model_secs=10 \
		--data_dir=dmo:///imagenet/tfrecord \
		--data_name=imagenet \
		--num_batches=500 \
		--train_dir=dmo:///resnet_model_dir

