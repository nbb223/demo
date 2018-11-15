#!/bin/bash

#PIPELINE_CONFIG_PATH="/memverge/home/songjue/data/ssd_inception_v2_coco_pascal.config"
PIPELINE_CONFIG_PATH="./ssd_inception_v2_coco_pascal_maui.config"
MODEL_DIR="/maui/scratch/songjue/model_dir"
NUM_TRAIN_STEPS=500
SAMPLE_1_OF_N_EVAL_EXAMPLES=100
#CUDA_VISIBLE_DEVICES=0,1 python /memverge/home/songjue/tensorflow/models/research/object_detection/model_main.py \
python /memverge/home/songjue/tensorflow/models/research/object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --save_checkpoints_steps=500 \
    --alsologtostder
    #--alsologtostder --num_gpus=2
