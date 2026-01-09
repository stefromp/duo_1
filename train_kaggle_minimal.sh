#!/bin/bash
# Minimal resource training - works on Kaggle T4 GPU (16GB)
# This is the most conservative setting that should work reliably

python -u main.py \
  --config-name=config_kaggle \
  loader.batch_size=4 \
  loader.eval_batch_size=4 \
  data=custom_local \
  model=tiny \
  model.length=256 \
  model.hidden_size=192 \
  model.n_blocks=6 \
  algo=duo_kaggle \
  algo.curriculum_start=2000 \
  algo.curriculum_end=8000 \
  trainer.max_steps=20000 \
  trainer.val_check_interval=2000 \
  trainer.accumulate_grad_batches=4 \
  trainer.precision='32' \
  training.ema=0.999 \
  eval.generate_samples=False \
  wandb.offline=true
