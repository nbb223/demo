#!/bin/bash

cd /memverge/home/songjue/tensorflow/models/official/resnet

### -rs,--resnet_size: <18|34|50|101|152|200>
### -rv,--resnet_version: <1|2>:
###  -mts,--max_train_steps


CUDA_VISIBLE_DEVICES=1 ~songjue/anaconda3/bin/python imagenet_main.py \
		--resnet_size=50 \
		--data_dir=hdfs://aep0:4545/imageNet/tfrecord \
		--model_dir=hdfs://aep0:4545/resNet_model_dir \
		--batch_size=32 \
	        --max_train_steps=500 \
		--clean=True \
		--train_epochs=1 \
		--undefok=--keep_checkpoint_max=5,--save_checkpoints_steps=10
