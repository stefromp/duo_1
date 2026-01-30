#!/bin/bash
#SBATCH -J duo-bimodal-owt            # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=64000                   # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=kuleshov          # Request partition
#SBATCH --constraint="[a5000|a6000|a100|3090]"
#SBATCH --constraint="gpu-mid|gpu-high"
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption

# =========================================================================
# DUO with Bimodal Gaussian Schedule + Frequency-Informed Training
# Based on: "Masked Diffusion Language Models with Frequency-Informed Training"
# (Kosmopoulou et al., 2025) - arXiv:2509.05056
#
# Key features:
# 1. Bimodal Gaussian noise schedule with time-varying right mode
# 2. Derivative softening (power=0.1) - CRITICAL for bimodal schedule
# 3. Frequency-informed masking: Prioritizes learning from rare tokens
# 4. Curriculum learning: Progressively focus on rare tokens
#
# From the paper: Bimodal Gaussian with derivative softening nearly reaches
# top baseline scores (Table 2).
# =========================================================================

srun python -u -m main \
  loader.batch_size=32 \
  loader.eval_batch_size=32 \
  data=openwebtext-split \
  wandb.name=duo-bimodal-owt \
  model=small \
  algo=duo_frequency_informed \
  noise=bimodal-gaussian \
  model.length=1024 \
  algo.gumbel_tau_log10_start=-3.0 \
  algo.gumbel_tau_log10_end=-3.0 \
  algo.gamma_min=-3.55 \
  algo.gamma_max=-1.85 \
  algo.curriculum_start=0 \
  algo.curriculum_end=500000 \
  algo.use_frequency_masking=True \
  algo.frequency_softening_power=0.02 \
  algo.frequency_curriculum=True \
  algo.derivative_power=0.1
