#!/bin/bash

# Train DUO on Kaggle with OpenWebText dataset
python -u main.py \
  --config-name=config_kaggle \
  loader.batch_size=8 \
  loader.eval_batch_size=8 \
  data=openwebtext \
  model=tiny \
  model.length=512 \
  algo=duo_kaggle \
  trainer.max_steps=50000
