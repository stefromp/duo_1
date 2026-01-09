#!/bin/bash
# Kaggle training script for DUO with custom dataset
# This script is optimized for single GPU training with limited resources

python -u main.py \
  --config-name=config_kaggle \
  loader.batch_size=8 \
  loader.eval_batch_size=8 \
  data=custom_local \
  wandb.name=duo-kaggle-custom \
  model=tiny \
  algo=duo_kaggle \
  model.length=512 \
  trainer.max_steps=50000 \
  trainer.val_check_interval=1000 \
  trainer.precision='32' \
  wandb.offline=false
