"""
Kaggle Training & Evaluation Notebook for DUO with Frequency-Informed Training
Based on: "Masked Diffusion Language Models with Frequency-Informed Training"
(Kosmopoulou et al., 2025) - arXiv:2509.05056

Run this as a Python script or copy cells into a Kaggle notebook.
"""

# =============================================================================
# CELL 1: Setup and Installation
# =============================================================================
print("=" * 60)
print("Setting up environment...")
print("=" * 60)

import subprocess
import sys

# Install requirements (run once)
# subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

import os
import torch
import numpy as np

# Check GPU
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}")
    print(f"Memory: {gpu_mem:.1f} GB")
    
    # Recommend batch size based on GPU
    if gpu_mem < 12:
        recommended_batch = 4
    elif gpu_mem < 20:
        recommended_batch = 8
    else:
        recommended_batch = 16
    print(f"Recommended batch size: {recommended_batch}")
else:
    print("WARNING: No GPU detected!")
    recommended_batch = 2

# =============================================================================
# CELL 2: Configuration
# =============================================================================
print("\n" + "=" * 60)
print("Configuration")
print("=" * 60)

class Config:
    # Model
    model_size = "tiny"      # "tiny", "small", "medium"
    seq_length = 128         # Sequence length (128 for testing, 512/1024 for full)
    
    # Training
    batch_size = 8           # Adjust based on GPU memory
    epochs = 10              # Paper uses 10 epochs
    learning_rate = 3e-4
    
    # Noise Schedule (from paper)
    # Options: "log-linear" (baseline), "cosine" (recommended), "bimodal-gaussian"
    noise_schedule = "cosine"
    
    # Frequency-Informed Masking (from paper)
    use_frequency_masking = True
    frequency_softening_power = 0.02  # Paper value
    frequency_curriculum = True        # Gradually increase focus on rare tokens
    
    # ELBO Derivative Softening
    # Use 1.0 for cosine/log-linear, 0.1 for bimodal-gaussian
    derivative_power = 1.0
    
    # Data
    data_dir = "./processed_data"
    
    # Output
    checkpoint_dir = "./checkpoints"
    log_dir = "./logs"

config = Config()
print(f"Model: {config.model_size}")
print(f"Batch size: {config.batch_size}")
print(f"Noise schedule: {config.noise_schedule}")
print(f"Frequency masking: {config.use_frequency_masking}")

# =============================================================================
# CELL 3: Training Function
# =============================================================================

def train_model(config):
    """Train DUO with frequency-informed masking."""
    
    cmd = [
        "python", "main.py",
        f"loader.batch_size={config.batch_size}",
        f"loader.eval_batch_size={config.batch_size}",
        "loader.num_workers=2",
        "data=custom_local",
        f"data.data_dir={config.data_dir}",
        f"model={config.model_size}",
        f"model.length={config.seq_length}",
        "algo=duo_frequency_informed",
        f"noise={config.noise_schedule}",
        f"algo.use_frequency_masking={config.use_frequency_masking}",
        f"algo.frequency_softening_power={config.frequency_softening_power}",
        f"algo.frequency_curriculum={config.frequency_curriculum}",
        f"algo.derivative_power={config.derivative_power}",
        "algo.curriculum_start=0",
        "algo.curriculum_end=5000",
        f"trainer.max_epochs={config.epochs}",
        "trainer.accelerator=gpu",
        "trainer.devices=1",
        "training.ema=0.9999",
        "eval.generate_samples=True",
        "eval.compute_generative_perplexity=False",  # Disable for faster training
        "wandb.mode=disabled",
        f"checkpointing.save_dir={config.checkpoint_dir}",
    ]
    
    print("\n" + "=" * 60)
    print("Starting training...")
    print("Command:", " ".join(cmd))
    print("=" * 60 + "\n")
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0

# =============================================================================
# CELL 4: Evaluation Function  
# =============================================================================

def evaluate_model(checkpoint_path, config):
    """Evaluate trained model on perplexity and generation quality."""
    
    cmd = [
        "python", "main.py",
        f"loader.batch_size={config.batch_size}",
        "data=custom_local",
        f"data.data_dir={config.data_dir}",
        f"model={config.model_size}",
        f"model.length={config.seq_length}",
        "algo=duo_frequency_informed",
        f"noise={config.noise_schedule}",
        f"algo.use_frequency_masking={config.use_frequency_masking}",
        f"algo.derivative_power={config.derivative_power}",
        "trainer.accelerator=gpu",
        "trainer.devices=1",
        "mode=eval",
        f"eval.checkpoint_path={checkpoint_path}",
        "eval.generate_samples=True",
        "eval.compute_generative_perplexity=True",
        "sampling.steps=256",
        "wandb.mode=disabled",
    ]
    
    print("\n" + "=" * 60)
    print("Starting evaluation...")
    print("=" * 60 + "\n")
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0

# =============================================================================
# CELL 5: Generate Samples
# =============================================================================

def generate_samples(checkpoint_path, num_samples=5, num_steps=256):
    """Generate text samples from trained model."""
    
    import torch
    import hydra
    from omegaconf import OmegaConf
    
    # This is a simplified example - adjust based on your main.py structure
    print(f"\nGenerating {num_samples} samples with {num_steps} steps...")
    print("Checkpoint:", checkpoint_path)
    
    # Load model (pseudo-code - adjust to your actual loading code)
    # model = load_checkpoint(checkpoint_path)
    # samples = model.generate_samples(num_samples, num_steps)
    # texts = tokenizer.batch_decode(samples)
    # for i, text in enumerate(texts):
    #     print(f"\n--- Sample {i+1} ---")
    #     print(text[:500])
    
    print("\n[Generate samples by loading checkpoint and calling model.generate_samples()]")

# =============================================================================
# CELL 6: Run Training
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DUO with Frequency-Informed Training")
    print("=" * 60)
    
    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Check data exists
    if not os.path.exists(config.data_dir):
        print(f"ERROR: Data directory not found: {config.data_dir}")
        print("Please ensure your training data is in the correct location.")
    else:
        print(f"Data directory: {config.data_dir}")
        print(f"Files: {os.listdir(config.data_dir)}")
    
    # =========================================================================
    # Option A: Run full training
    # =========================================================================
    # train_model(config)
    
    # =========================================================================
    # Option B: Run with different configurations for ablation
    # =========================================================================
    experiments = [
        # Baseline: Standard DUO (log-linear, no frequency masking)
        {"noise_schedule": "log-linear", "use_frequency_masking": False, "name": "baseline"},
        
        # Cosine schedule only
        {"noise_schedule": "cosine", "use_frequency_masking": False, "name": "cosine"},
        
        # Frequency masking only
        {"noise_schedule": "log-linear", "use_frequency_masking": True, "name": "freq_mask"},
        
        # Full: Cosine + Frequency masking (recommended)
        {"noise_schedule": "cosine", "use_frequency_masking": True, "name": "full"},
        
        # Bimodal Gaussian (requires derivative_power=0.1)
        {"noise_schedule": "bimodal-gaussian", "use_frequency_masking": True, 
         "derivative_power": 0.1, "name": "bimodal"},
    ]
    
    print("\n" + "=" * 60)
    print("Available experiment configurations:")
    print("=" * 60)
    for i, exp in enumerate(experiments):
        print(f"  {i}: {exp['name']}")
    
    print("\nTo run an experiment, uncomment and modify the training code above.")
    print("Example:")
    print("  config.noise_schedule = 'cosine'")
    print("  config.use_frequency_masking = True")
    print("  train_model(config)")

# =============================================================================
# CELL 7: Quick Test (Sanity Check)
# =============================================================================

def quick_test():
    """Run a quick test to verify setup."""
    print("\n" + "=" * 60)
    print("Running quick sanity check...")
    print("=" * 60)
    
    # Test imports
    try:
        import frequency_masking
        print("✓ frequency_masking module imported")
        
        import trainer_base
        print("✓ trainer_base module imported")
        
        # Test noise schedules
        t = torch.tensor([0.1, 0.5, 0.9])
        
        cosine = trainer_base.CosineSchedule()
        dalpha, alpha = cosine(t)
        print(f"✓ CosineSchedule: α(0.5) = {alpha[1]:.4f}")
        
        bimodal = trainer_base.BimodalGaussianSchedule()
        dalpha, alpha = bimodal(t)
        print(f"✓ BimodalGaussianSchedule: α(0.5) = {alpha[1]:.4f}")
        
        # Test frequency masking
        freq_mask = frequency_masking.FrequencyInformedMasking(
            vocab_size=1000,
            softening_power=0.02,
        )
        print("✓ FrequencyInformedMasking created")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

# Uncomment to run:
# quick_test()
