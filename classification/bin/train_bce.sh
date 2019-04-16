#!/usr/bin/env bash

echo "Train..."
CUDA_VISIBLE_DEVICES="${GPUS}" \
    catalyst-dl run \
    --config="classification/configs/train.yml" \
    --stages/criterion_params/criterion=BCEWithLogitsLoss:str \
    --verbose
