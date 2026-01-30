#!/bin/bash
# =============================================================================
# Kaggle training script for DUO with Frequency-Informed Training
# Based on: "Masked Diffusion Language Models with Frequency-Informed Training"
# (Kosmopoulou et al., 2025) - arXiv:2509.05056
#
# This script is optimized for single GPU training (Kaggle P100/T4)
# =============================================================================

# Choose training mode:
# 1. BASELINE - Standard DUO (for comparison)
# 2. COSINE - Cosine schedule only (quick improvement)  
# 3. FREQUENCY - Frequency masking + Cosine (recommended)
# 4. BIMODAL - Bimodal Gaussian + Frequency (best potential, needs tuning)

MODE=${1:-"FREQUENCY"}  # Default to FREQUENCY mode

case $MODE in
  "BASELINE")
    echo "Training BASELINE DUO (log-linear, no frequency masking)"
    NOISE="log-linear"
    USE_FREQ="False"
    DERIV_POWER="1.0"
    ;;
  "COSINE")
    echo "Training with COSINE schedule only"
    NOISE="cosine"
    USE_FREQ="False"
    DERIV_POWER="1.0"
    ;;
  "FREQUENCY")
    echo "Training with COSINE + FREQUENCY masking (recommended)"
    NOISE="cosine"
    USE_FREQ="True"
    DERIV_POWER="1.0"
    ;;
  "BIMODAL")
    echo "Training with BIMODAL Gaussian + FREQUENCY masking"
    NOISE="bimodal-gaussian"
    USE_FREQ="True"
    DERIV_POWER="0.1"  # Critical for bimodal!
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Usage: $0 [BASELINE|COSINE|FREQUENCY|BIMODAL]"
    exit 1
    ;;
esac

python -u main.py \
  --config-name=config_kaggle \
  loader.batch_size=8 \
  loader.eval_batch_size=8 \
  loader.num_workers=2 \
  data=custom_local \
  wandb.name=duo-kaggle-${MODE,,} \
  model=tiny \
  algo=duo_kaggle \
  noise.type=${NOISE} \
  model.length=512 \
  algo.use_frequency_masking=${USE_FREQ} \
  algo.frequency_softening_power=0.02 \
  algo.frequency_curriculum=True \
  algo.derivative_power=${DERIV_POWER} \
  trainer.max_steps=50000 \
  trainer.val_check_interval=1000 \
  trainer.precision='32' \
  eval.generate_samples=True \
  wandb.offline=false
