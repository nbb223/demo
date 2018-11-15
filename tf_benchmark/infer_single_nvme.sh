#!/bin/bash

#TRAIN_DIR=/tmp/resNet-model-dir
TRAIN_DIR=/memverge/home/songjue/resnet50-model-dir222
rm -f  ${TRAIN_DIR}/*

if [ $(hostname)  == 'maui' ]; then
    export CUDA_VISIBLE_DEVICES=0
    DATA_DIR='/maui/scratch/share/imagenet_tfrecord'
elif [ $(hostname)  == 'bigisland' ]; then
    export CUDA_VISIBLE_DEVICES=1
    DATA_DIR='/home/yli/nvme_ssd/imagenet/tfrecord'
fi

cd ~songjue/tensorflow/benchmarks/scripts/tf_cnn_benchmarks   

~songjue/anaconda3-gpu/bin/python tf_cnn_benchmarks.py \
		--forward_only=true \
                --save_model_steps=0 \
		--num_gpus=1 \
		--batch_size=512 \
		--model=resnet50 \
		--variable_update=parameter_server \
		--data_dir=${DATA_DIR} \
		--num_batches=500 \
		--train_dir=${TRAIN_DIR}
