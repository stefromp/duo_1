#!/bin/bash
# Standard Kaggle training - uses more resources but trains faster
# Recommended for T4 GPU with 16GB

python -u main.py \
  --config-name=config_kaggle \
  loader.batch_size=8 \
  loader.eval_batch_size=8 \
  data=custom_local \
  model=tiny \
  model.length=512 \
  algo=duo_kaggle \
  algo.curriculum_start=5000 \
  algo.curriculum_end=15000 \
  training.ema=0.9999 \
  eval.generate_samples=False \
  wandb.offline=false
