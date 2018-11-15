#!/bin/bash

echo "*** Usage: stributed_single.sh <ps|wk> TASK_IDX SHARD_NUM"

export OMP_NUM_THREADS=9 
export TF_ADJUST_HUE_FUSED=1
export TF_ADJUST_SATURATION_FUSED=1

TASK_INDEX=$2
SHARD_IDX=$2
SHARD_NUM=$3

if [ $1 == 'ps' ]; then
  JOB_NAME='ps'
  PARAM_DEV='cpu'
  EXTRA_ARGS='--num_intra_threads 4 --num_inter_threads 2'
  CTL='numactl -l' 
elif [ $1 == 'wk' ]; then
  JOB_NAME='worker'
  PARAM_DEV='gpu'
  EXTRA_ARGS='--num_intra_threads 9 --num_inter_threads 4'
   if [ $2 == '0' -o $2 == '1' ]; then
      CTL='numactl -m 0 '
   else 
      CTL='numactl -m 1 '
   fi
else
  echo "Err: job_name must be one of <ps|wk>"
  exit -1
fi

if [ $(hostname)  == 'maui' ]; then
    DATA_DIR='/maui/scratch/share/imagenet_tfrecord'
elif [ $(hostname)  == 'bigisland' ]; then
    DATA_DIR='/home/yli/nvme_ssd/imagenet/tfrecord'
fi

MODEL='resnet50'
SAVE_MODEL_STEPS=100
#WORKERS='maui:5432,bigisland:5432'
WORKERS='maui:5432,maui:5442,maui:5452,maui:5462'
PS='bigisland:6543'

export CUDA_VISIBLE_DEVICES=''   #use cpu only

cd ~songjue/tensorflow/benchmarks/scripts/tf_cnn_benchmarks

# --sync_on_finish: Enable/disable whether the devices are synced after each step.
# --use_fp16

$CTL ~songjue/anaconda3-cpu/bin/python tf_cnn_benchmarks.py \
                ${EXTRA_ARGS} \
                --mkl=True \
                --optimizer=rmsprop \
                --data_format=NHWC \
                --device='cpu' \
                --cache_data=True \
                --num_shards=${SHARD_NUM} \
                -shard_idx=${SHARD_IDX} \
		--task_index=${TASK_INDEX} \
                --use_fp16=True \
                --fp16_vars=True \
		--job_name=${JOB_NAME} \
                --ps_hosts=${PS}\
                --worker_hosts=${WORKERS} \
		--local_parameter_device=cpu  \
		--save_model_steps=${SAVE_MODEL_STEPS} \
		--batch_size=64 \
		--model=${MODEL} \
		--data_dir=${DATA_DIR} \
		--data_name=imagenet \
		--num_batches=1000 
