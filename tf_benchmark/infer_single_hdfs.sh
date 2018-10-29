#!/bin/bash

#### run as user 'dmo' ####
MODEL=resnet50

export HADOOP_HDFS_HOME=/memverge/home/songjue/hadoop-3.1.1
export CLASSPATH=$CLASSPATH:$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob)

if [ $(hostname)  == 'maui' ]; then
    export CUDA_VISIBLE_DEVICES=0
elif [ $(hostname)  == 'bigisland' ]; then
    export CUDA_VISIBLE_DEVICES=1
fi

cd ~songjue/tensorflow/benchmarks/scripts/tf_cnn_benchmarks
~songjue/anaconda3-gpu/bin/python tf_cnn_benchmarks.py \
		--forward_only=true \
                --save_model_steps=0 \
		--num_gpus=1 \
		--batch_size=64 \
		--model=${MODEL} \
		--variable_update=parameter_server \
		--data_dir=hdfs://aep0:4545/imageNet/tfrecord \
		--data_name=imagenet \
                --num_batches=500 
#		--train_dir=hdfs://aep0:4545/${MODEL}_model_dir
		#--train_dir=/tmp/reNet-model-dir
		#--train_dir=/home/yli/nvme_ssd/songjue/renet_model_dir 

