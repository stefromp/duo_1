"""
Kaggle Notebook Setup for DUO Training
Copy and paste these cells into your Kaggle notebook
"""

# ============================================================================
# CELL 1: Clone Repository (Dataset included - 79 MB)
# ============================================================================
!git clone https://github.com/stefromp/duo_1.git
%cd duo_1

# ============================================================================
# CELL 2: Install Dependencies
# ============================================================================
!pip install -q -r requirements.txt

# Note: Flash Attention is NOT needed!
# PyTorch 2.3+ includes optimized SDPA (Scaled Dot Product Attention)
# which is almost as fast and requires no compilation.
# If you want Flash Attention anyway (optional, takes 10+ minutes):
# !pip install -q flash-attn --no-build-isolation

# ============================================================================
# CELL 3: Verify Setup (Dataset already included in repo!)
# ============================================================================
import os

# Check GPU
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Check dataset
dataset_path = "processed_data /train_subset_clean.txt"
if os.path.exists(dataset_path):
    line_count = sum(1 for _ in open(dataset_path, 'rb'))
    print(f"✓ Dataset found: {line_count:,} lines")
else:
    print(f"✗ Dataset not found at: {dataset_path}")

# Check integral cache
cache_path = "integral/gpt2.pkl"
if os.path.exists(cache_path):
    print(f"✓ Integral cache found")
else:
    print(f"✗ Generating integral cache...")
    !python utils.py --vocab_size=50257
    print(f"✓ Integral cache generated")

print("\n=== Setup Complete! ===")

# ============================================================================
# CELL 4: Start Training - Minimal Resources (RECOMMENDED FOR FIRST RUN)
# ============================================================================
# This uses the most conservative settings
# Memory usage: ~8-10 GB
# Training time: ~10 hours for 20k steps

!python -u main.py \
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

# ============================================================================
# CELL 4 (ALTERNATIVE): Standard Training - More Resources
# ============================================================================
# Use this if minimal works well and you want faster training
# Memory usage: ~12-14 GB
# Training time: ~6-8 hours for 50k steps

"""
!python -u main.py \
  --config-name=config_kaggle \
  loader.batch_size=8 \
  loader.eval_batch_size=8 \
  data=custom_local \
  model=tiny \
  model.length=512 \
  algo=duo_kaggle \
  trainer.max_steps=50000 \
  trainer.val_check_interval=1000 \
  wandb.offline=false
"""

# ============================================================================
# CELL 5: Monitor Training (Run in separate cell while training)
# ============================================================================
"""
# Check GPU memory usage
!nvidia-smi

# Check latest checkpoint
!ls -lht outputs/custom/*/checkpoints/ | head -5

# View recent logs (if saved)
!tail -50 outputs/custom/*/train.log
"""

# ============================================================================
# CELL 6: Generate Samples After Training
# ============================================================================
"""
# Replace with your actual checkpoint path
checkpoint_path = "outputs/custom/2024.01.09/123456/checkpoints/last.ckpt"

!python main.py \
  mode=sample_eval \
  eval.checkpoint_path={checkpoint_path} \
  sampling.steps=50 \
  sampling.num_sample_batches=2 \
  loader.eval_batch_size=4 \
  wandb.offline=true
"""

# ============================================================================
# CELL 7: Save Checkpoints
# ============================================================================
"""
# Create a zip file of all checkpoints
!zip -r checkpoints.zip outputs/

# In Kaggle, this will be saved to /kaggle/working/
# You can download it from the Output section

# Or copy to specific location
!mkdir -p /kaggle/working/saved_models
!cp outputs/custom/*/checkpoints/last.ckpt /kaggle/working/saved_models/
"""

# ============================================================================
# CELL 8: Resume Training (if session disconnected)
# ============================================================================
"""
# Find your checkpoint path
!find outputs -name "*.ckpt" -type f

# Resume from checkpoint
checkpoint_path = "outputs/custom/2024.01.09/123456/checkpoints/last.ckpt"

!python main.py \
  --config-name=config_kaggle \
  checkpointing.resume_from_ckpt=true \
  checkpointing.resume_ckpt_path={checkpoint_path} \
  trainer.max_steps=100000
"""

# ============================================================================
# TROUBLESHOOTING
# ============================================================================
"""
# If you get Out of Memory error:
# 1. Reduce batch size in CELL 5:
#    loader.batch_size=2
#    loader.global_batch_size=8

# 2. Reduce sequence length:
#    model.length=128

# 3. Reduce model size:
#    model.hidden_size=128
#    model.n_blocks=4

# If training is too slow:
# 1. Reduce validation frequency:
#    trainer.val_check_interval=5000

# 2. Disable sample generation:
#    eval.generate_samples=False

# 3. Use smaller validation set:
#    trainer.limit_val_batches=0.05

# If you see NaN loss:
# 1. Lower learning rate:
#    optim.lr=1e-4

# 2. Check your dataset for corrupted data
# 3. Try reducing batch size
"""
