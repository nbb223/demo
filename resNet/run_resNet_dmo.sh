#!/bin/bash

cd /memverge/home/songjue/tensorflow/models/official/resnet

### -rs,--resnet_size: <18|34|50|101|152|200>
### -rv,--resnet_version: <1|2>:
###  -mts,--max_train_steps


CUDA_VISIBLE_DEVICES=1 ~songjue/anaconda3-gpu/bin/python imagenet_main.py \
		--enable_dmo \
		--resnet_size=50 \
		--data_dir=dmo:///imagenet/tfrecord \
		--model_dir=dmo:///resnet_model_dir \
		--batch_size=32 \
	        --max_train_steps=500 \
		--clean=True \
		--train_epochs=1 \
		--undefok=--keep_checkpoint_max=5,--save_checkpoints_steps=10
