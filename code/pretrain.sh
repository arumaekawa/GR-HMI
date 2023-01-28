#!/bin/bash -i

source ./env

python pretrain.py \
  --pretrain_data_dir $PRETRAIN_DATA_DIR \
  --pretrained_model_dir_root $PRETRAINED_MODEL_ROOT_DIR \
  "$@"
