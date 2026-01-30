# DUO with Frequency-Informed Training - Kaggle Guide

Based on: **"Masked Diffusion Language Models with Frequency-Informed Training"**  
(Kosmopoulou et al., 2025) - [arXiv:2509.05056](https://arxiv.org/abs/2509.05056)

---

## 📊 Paper Summary

This paper introduces three key techniques to improve masked diffusion language models:

| Technique | Description | Improvement |
|-----------|-------------|-------------|
| **Frequency-Informed Masking** | Rare tokens are masked more often | +1-7% on morphology/BLiMP |
| **Cosine Noise Schedule** | α_t = cos(π/2 × (1-t)) | +1-3% over linear |
| **Bimodal Gaussian Schedule** | Two-mode distribution with curriculum | Near top baseline (with softening) |
| **ELBO Derivative Softening** | Scale derivative by power < 1 | Critical for bimodal schedule |

---

## 🔧 Implementation Summary

### Files Changed

| File | Status | Description |
|------|--------|-------------|
| `frequency_masking.py` | **NEW** | Core frequency masking + noise schedules |
| `trainer_base.py` | Modified | Integration with training loop |
| `configs/algo/duo_kaggle.yaml` | Modified | Added frequency params |
| `configs/algo/duo_frequency_informed.yaml` | **NEW** | Full frequency config |
| `configs/noise/cosine.yaml` | **NEW** | Cosine schedule config |
| `configs/noise/bimodal-gaussian.yaml` | **NEW** | Bimodal schedule config |
| `train_kaggle.sh` | Modified | Multi-mode training script |
| `train_kaggle_frequency.sh` | **NEW** | Frequency-focused training |
| `kaggle_train_evaluate.py` | **NEW** | Full Kaggle notebook script |

### Key Configuration Options

```yaml
# In algo config (e.g., duo_kaggle.yaml)
algo:
  use_frequency_masking: True      # Enable frequency masking
  frequency_softening_power: 0.02  # Softening (paper value)
  frequency_curriculum: True       # Gradually increase focus
  derivative_power: 1.0            # Use 0.1 for bimodal schedule

# In noise config
noise:
  type: cosine  # Options: log-linear, cosine, bimodal-gaussian
```

---

## 🚀 Kaggle Training Guide

### Step 1: Upload to Kaggle

1. Create a new Kaggle notebook
2. Upload your code or clone from GitHub:

```python
!git clone https://github.com/stefromp/duo_1.git
%cd duo_1
!pip install -r requirements.txt
```

### Step 2: Choose Training Mode

The updated `train_kaggle.sh` supports 4 modes:

```bash
# Option 1: BASELINE - Standard DUO (for comparison)
!bash train_kaggle.sh BASELINE

# Option 2: COSINE - Cosine schedule only (+1-3% improvement)
!bash train_kaggle.sh COSINE

# Option 3: FREQUENCY - Cosine + Frequency masking (RECOMMENDED)
!bash train_kaggle.sh FREQUENCY

# Option 4: BIMODAL - Bimodal Gaussian + Frequency (best potential)
!bash train_kaggle.sh BIMODAL
```

### Step 3: Or Run Python Directly

```python
!python main.py \
  loader.batch_size=8 \
  data=custom_local \
  model=tiny \
  algo=duo_kaggle \
  noise.type=cosine \
  algo.use_frequency_masking=True \
  algo.frequency_softening_power=0.02 \
  algo.frequency_curriculum=True \
  trainer.max_steps=50000 \
  wandb.mode=disabled
```

### Step 4: Evaluate

```python
!python main.py \
  mode=eval \
  eval.checkpoint_path="./checkpoints/last.ckpt" \
  eval.generate_samples=True \
  eval.compute_generative_perplexity=True \
  sampling.steps=256
```

---

## 📈 Expected Results

Based on paper's experiments (Table 3):

| Configuration | BLiMP % | BLiMP Sup. % | Notes |
|---------------|---------|--------------|-------|
| Baseline (log-linear) | 77.9 | 67.6 | Standard DUO |
| + Cosine schedule | 79.0 | 70.7 | +1.1% / +3.1% |
| + Frequency masking | 78.9 | 71.8 | +1% on supplement |
| Bimodal + softening(0.1) | 79.5 | 72.8 | Best overall |

---

## ⚙️ Recommended Settings for Kaggle

### GPU Memory Guide

| GPU | VRAM | Batch Size | Sequence Length |
|-----|------|------------|-----------------|
| P100 | 16GB | 8 | 512 |
| T4 | 16GB | 8 | 512 |
| V100 | 32GB | 16 | 1024 |

### Quick Start (Conservative)

```bash
!python main.py \
  loader.batch_size=4 \
  model=tiny \
  model.length=128 \
  algo=duo_kaggle \
  noise.type=cosine \
  algo.use_frequency_masking=True \
  trainer.max_steps=10000 \
  wandb.mode=disabled
```

### Full Training (Recommended)

```bash
!python main.py \
  loader.batch_size=8 \
  model=tiny \
  model.length=512 \
  algo=duo_kaggle \
  noise.type=cosine \
  algo.use_frequency_masking=True \
  algo.frequency_softening_power=0.02 \
  algo.frequency_curriculum=True \
  trainer.max_steps=50000 \
  trainer.val_check_interval=2500 \
  wandb.mode=disabled
```

---

## 🔬 Ablation Experiments

Run these to understand each component's contribution:

```python
experiments = {
    "baseline": "noise.type=log-linear algo.use_frequency_masking=False",
    "cosine_only": "noise.type=cosine algo.use_frequency_masking=False", 
    "freq_only": "noise.type=log-linear algo.use_frequency_masking=True",
    "cosine+freq": "noise.type=cosine algo.use_frequency_masking=True",
    "bimodal+freq": "noise.type=bimodal-gaussian algo.use_frequency_masking=True algo.derivative_power=0.1",
}

for name, args in experiments.items():
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}\n")
    !python main.py {args} trainer.max_steps=10000 wandb.name={name}
```

---

## ⚠️ Important Notes

1. **Bimodal Gaussian requires derivative softening**
   ```yaml
   algo.derivative_power: 0.1  # CRITICAL for bimodal!
   ```
   Without this, bimodal performs ~10% worse.

2. **Curriculum learning starts at epoch 0**
   - Frequency masking gradually increases focus on rare tokens
   - Bimodal schedule's right mode moves from 0.4 → 0.85 during training

3. **Memory optimization**
   - Reduce `model.length` first (128 → 512 → 1024)
   - Then reduce `batch_size` if needed
   - Use gradient accumulation for effective larger batches

4. **Checkpoint saving**
   ```yaml
   checkpointing.save_dir: "./checkpoints"
   checkpointing.every_n_steps: 5000
   ```

---

## 📁 Project Structure After Changes

```
duo-main/
├── frequency_masking.py          # NEW: Core frequency masking
├── trainer_base.py               # MODIFIED: Integration
├── configs/
│   ├── algo/
│   │   ├── duo_kaggle.yaml       # MODIFIED: Added frequency params
│   │   └── duo_frequency_informed.yaml  # NEW
│   └── noise/
│       ├── log-linear.yaml       # Existing
│       ├── cosine.yaml           # NEW
│       └── bimodal-gaussian.yaml # NEW
├── scripts/
│   ├── train_owt_duo_frequency.sh  # NEW
│   └── train_owt_duo_bimodal.sh    # NEW
├── train_kaggle.sh               # MODIFIED: Multi-mode
├── train_kaggle_frequency.sh     # NEW
└── kaggle_train_evaluate.py      # NEW: Full Kaggle script
```

---

## 🎯 Quick Commands Reference

```bash
# Standard training with all improvements
!bash train_kaggle.sh FREQUENCY

# Or use the detailed frequency script
!bash train_kaggle_frequency.sh

# Or run the Python evaluation script
!python kaggle_train_evaluate.py
```

---

## 📚 Citation

If you use these improvements, please cite:

```bibtex
@article{kosmopoulou2025masked,
  title={Masked Diffusion Language Models with Frequency-Informed Training},
  author={Kosmopoulou, Despoina and Georgiou, Efthymios and Dorovatas, Vaggelis 
          and Paraskevopoulos, Georgios and Potamianos, Alexandros},
  journal={arXiv preprint arXiv:2509.05056},
  year={2025}
}
```
