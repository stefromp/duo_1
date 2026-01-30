#!/bin/bash
# =============================================================================
# Kaggle Training Script for DUO with Frequency-Informed Training
# Based on: "Masked Diffusion Language Models with Frequency-Informed Training"
# (Kosmopoulou et al., 2025) - arXiv:2509.05056
# =============================================================================
#
# Usage in Kaggle notebook:
#   !bash train_kaggle_frequency.sh
#
# Or run Python directly:
#   !python main.py [args from below]
#
# Kaggle GPUs:
#   - P100: 16GB VRAM → batch_size=8-16
#   - T4: 16GB VRAM → batch_size=8-16  
#   - GPU quota: ~30 hours/week
#
# Training time estimates (T4/P100, 10 epochs):
#   - tiny model: ~2-3 hours
#   - small model: ~6-8 hours
# =============================================================================

# Configuration
MODEL_SIZE="tiny"              # tiny, small, medium
BATCH_SIZE=8                   # Adjust based on GPU memory
EPOCHS=10                      # Paper uses 10 epochs
NOISE_SCHEDULE="cosine"        # cosine, log-linear, bimodal-gaussian
USE_FREQ_MASKING=True          # Enable frequency-informed masking
FREQ_POWER=0.02                # Softening power for frequency weights
DERIV_POWER=1.0                # Use 0.1 for bimodal-gaussian schedule

# Dataset - use your local processed data or openwebtext
DATA_CONFIG="custom_local"     # or openwebtext-split

python main.py \
  loader.batch_size=${BATCH_SIZE} \
  loader.eval_batch_size=${BATCH_SIZE} \
  loader.num_workers=2 \
  data=${DATA_CONFIG} \
  data.data_dir="./processed_data" \
  model=${MODEL_SIZE} \
  model.length=128 \
  algo=duo_frequency_informed \
  noise=${NOISE_SCHEDULE} \
  algo.use_frequency_masking=${USE_FREQ_MASKING} \
  algo.frequency_softening_power=${FREQ_POWER} \
  algo.frequency_curriculum=True \
  algo.derivative_power=${DERIV_POWER} \
  algo.curriculum_start=0 \
  algo.curriculum_end=5000 \
  trainer.max_epochs=${EPOCHS} \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  training.ema=0.9999 \
  eval.generate_samples=True \
  eval.compute_generative_perplexity=True \
  wandb.mode=disabled \
  checkpointing.save_dir="./checkpoints"
