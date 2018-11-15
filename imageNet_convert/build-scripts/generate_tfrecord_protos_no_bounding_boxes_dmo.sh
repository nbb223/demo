#!/bin/bash
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

set -e

DATA_ROOT="dmo:///imagenet"

DATA_DIR=$DATA_ROOT/ilsvrc2012
PROTO_OUT_DIR=$DATA_ROOT/tfrecord
RESIZE_IMAGES="false"
NEW_HEIGHT="352"
NEW_WIDTH="352"
JPEG_Q="100"

if [ "x$PROTO_OUT_DIR" == x ]; then
cat <<END
 Generate TFRecord protobufs. You can either set env vars such as
 IMAGENET_DATA_DIR, or enter them here.
END
  read -p "ImageNet data dir: " DATA_DIR
  read -p "Proto out dir:  " PROTO_OUT_DIR
  read -p "Pre-resize images to uniform size (Y/n): " RESIZE_IMAGES
  if [ "${RESIZE_IMAGES,,}" = "y" ]; then
    echo "We will need some information from you about resizing"
    read -p "Image height after resizing (in pixels, recommended 352): " NEW_HEIGHT
    read -p "Image width after resizing (in pixels, recommended 352): " NEW_WIDTH
    read -p "JPEG Q after resizing (0 to 100, recommended 90): " JPEG_Q
  else
    RESIZE_IMAGES="false"
    NEW_HEIGHT="352"
    NEW_WIDTH="352"
    JPEG_Q="90"
  fi
fi

BUILD_SCRIPT="build_imagenet_data_dmo.py"
IMAGENET_META_FILE="imagenet_metadata.txt"
LABELS_FILE="imagenet_lsvrc_2015_synsets.txt"
TRAIN_DIR="${DATA_DIR}/data/train"
VALIDATION_DIR="${DATA_DIR}/data/validation"
#BBOX_FILE="${DATA_DIR}/bounding_boxes/imagenet_2012_bounding_boxes.csv"

#echo "Bounding boxes at ${BBOX_FILE}"

CUDA_VISIBLE_DEVICES=1  ~songjue/anaconda3-cpu/bin/python "${BUILD_SCRIPT}" \
  --num_threads=32 \
  --validation_shards=128 \
  --train_directory="${TRAIN_DIR}" \
  --validation_directory="${VALIDATION_DIR}" \
  --output_directory="${PROTO_OUT_DIR}" \
  --imagenet_metadata_file="${IMAGENET_META_FILE}" \
  --labels_file="${LABELS_FILE}" \
  --resize_images="${RESIZE_IMAGES}" \
  --new_height="${NEW_HEIGHT}" \
  --new_width="${NEW_WIDTH}" \
  --jpeg_q="${JPEG_Q}"
