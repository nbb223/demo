#!/bin/bash

if [ $1 == 'ps' ]; then
  JOB_NAME='ps'
  PARAM_DEV='cpu'
elif [ $1 == 'wk' ]; then
  JOB_NAME='worker'
  PARAM_DEV='gpu'
else
  echo "Err: job_name must be one of <ps|wk>"
  exit -1
fi

MODEL='resnet50'
SAVE_MODEL_STEPS=10
#WORKERS='maui:5432,bigisland:5432'
WORKERS='bigisland:5432,maui:5432'
PS='bigisland:6543'

cd ~songjue/tensorflow/benchmarks/scripts/tf_cnn_benchmarks

TASK_INDEX=1
if [ $(hostname)  == 'maui' ]; then
    export CUDA_VISIBLE_DEVICES=0
elif [ $(hostname)  == 'bigisland' ]; then
    export CUDA_VISIBLE_DEVICES=1
    TASK_INDEX=0
fi

if [ ${JOB_NAME} == 'ps' ]; then
    export CUDA_VISIBLE_DEVICES=''
   TASK_INDEX=0
fi

# --sync_on_finish: Enable/disable whether the devices are synced after each step.
# --use_fp16

~songjue/anaconda3-gpu/bin/python tf_cnn_benchmarks.py \
		--enable_dmo \
		--task_index=${TASK_INDEX} \
		--job_name=${JOB_NAME} \
                --ps_hosts=${PS}\
                --worker_hosts=${WORKERS} \
		--local_parameter_device=${PARAM_DEV} \
		--save_model_steps=${SAVE_MODEL_STEPS} \
		--batch_size=32 \
		--model=${MODEL} \
		--variable_update=distributed_replicated \
		--data_dir=dmo:///imagenet/tfrecord \
		--data_name=imagenet \
		--num_batches=500 \
		--train_dir=dmo:///${MODEL}_model_dir
