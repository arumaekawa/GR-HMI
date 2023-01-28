#!/bin/bash -i

source ./env

python train.py \
  --data_dir $CL_DATA_DIR \
  --model_dir_root $CL_MODEL_ROOT_DIR \
  "$@"
