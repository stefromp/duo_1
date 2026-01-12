# 🚀 Kaggle Setup Instructions - CRITICAL FIX

## ⚠️ PROBLEM: Git LFS Budget Exceeded

Your GitHub repo exceeded LFS bandwidth limit. The data files (50MB+) cannot be cloned.

## ✅ SOLUTION: Upload Data Directly to Kaggle

### Step 1: Create Kaggle Dataset

1. **Download your data files locally** (you already have them):
   - `processed_data/train_subset_clean.txt` (48MB)
   - `processed_data/val_subset_clean.txt` (4.8MB)

2. **Go to Kaggle Datasets**: https://www.kaggle.com/datasets

3. **Click "New Dataset"**

4. **Upload Files**:
   - Drag and drop both `.txt` files
   - Or click "Upload" and select them

5. **Set Dataset Info**:
   - **Title**: `duo-custom-training-data`
   - **Slug**: `duo-custom-training-data` (auto-generated)
   - **Subtitle**: "Training and validation data for DUO model"
   - **Description**: "Fixed 90/10 split of general web text for language modeling"
   - **Visibility**: Private (or Public if you want)

6. **Click "Create"**

7. **Copy the dataset path**: It will be something like:
   ```
   /kaggle/input/duo-custom-training-data/
   ```

---

### Step 2: Update Code to Skip Data Files

You need to clone the repo WITHOUT the large data files, then mount the Kaggle dataset instead.

---

## 🎯 UPDATED KAGGLE TRAINING CELL

Use this code in your Kaggle notebook:

```python
# ═══════════════════════════════════════════════════════════
# KAGGLE TRAINING SETUP - WITH EXTERNAL DATASET
# ═══════════════════════════════════════════════════════════

import os

# Go to working directory
%cd /kaggle/working

# Clean slate
!rm -rf duo_1

# Clone repo (will fail on data files - that's OK!)
!git clone https://github.com/stefromp/duo_1.git 2>&1 | grep -v "smudge filter" || true
%cd duo_1

# Fix the checkout by skipping LFS
!git lfs install --skip-smudge
!git reset --hard HEAD

# Create processed_data directory
!mkdir -p processed_data

# Copy data from Kaggle dataset (CHANGE THIS PATH!)
# After you create your dataset, add it to your notebook:
# 1. Click "Add Data" in Kaggle notebook
# 2. Search for "duo-custom-training-data" 
# 3. Click "Add"
# 4. The data will be at: /kaggle/input/duo-custom-training-data/

!cp /kaggle/input/duo-custom-training-data/train_subset_clean.txt processed_data/
!cp /kaggle/input/duo-custom-training-data/val_subset_clean.txt processed_data/

# Verify data is there
!ls -lh processed_data/
!wc -l processed_data/*.txt

# Install requirements (retry on timeout)
!pip install -q -r requirements.txt || pip install -q -r requirements.txt

# Verify installation
!python -c "import hydra; print('✅ Hydra installed')"

# ═══════════════════════════════════════════════════════════
# START TRAINING - SMALL MODEL (RECOMMENDED)
# ═══════════════════════════════════════════════════════════

!python -u main.py \
  --config-name=config_kaggle \
  loader.batch_size=8 \
  loader.eval_batch_size=8 \
  data=custom_local \
  model=small \
  model.length=512 \
  algo=duo_kaggle \
  trainer.max_steps=15000 \
  trainer.val_check_interval=2000 \
  trainer.log_every_n_steps=100 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=1500 \
  optim.lr=3e-4 \
  training.ema=0.9999 \
  trainer.precision=bf16
```

---

## 🔧 ALTERNATIVE: Remove LFS from GitHub (Long-term Fix)

If you want to fix your GitHub repo permanently:

```bash
# On your local machine
cd /Users/stefanosrompos/Desktop/duo-main

# Remove files from LFS tracking
git lfs untrack "processed_data/*.txt"

# Remove LFS version and commit regular files
git rm --cached processed_data/*.txt
git add processed_data/*.txt
git add .gitattributes

# Commit
git commit -m "Remove LFS tracking - files will be in Kaggle dataset instead"

# Force push (WARNING: This rewrites history!)
# git push origin main --force

# Or just remove the files entirely from repo
git rm processed_data/*.txt
git commit -m "Remove data files - use Kaggle dataset instead"
git push origin main
```

Then add a README note that data should come from Kaggle dataset.

---

## 📊 GPU SELECTION

**For P100 GPU on Kaggle:**

1. Click the 3 dots (⋮) in top right of notebook
2. Click "Notebook options" or "Session options"
3. Under "Accelerator", select **"GPU P100"**
4. Click "Save"
5. Your notebook will restart with P100

**Why P100?**
- ✅ 30-40% faster than T4
- ✅ Same 16GB VRAM
- ✅ Free tier access
- ✅ Saves 2-3 hours on training

---

## ✅ CHECKLIST

Before running training:

- [ ] Create Kaggle dataset with your data files
- [ ] Add dataset to your Kaggle notebook (click "Add Data")
- [ ] Update the dataset path in the code above
- [ ] Select GPU P100 in notebook settings
- [ ] Run the updated training cell
- [ ] Monitor for first validation checkpoint

Expected training time: **5-6 hours on P100** for 15k steps

---

## 🆘 If You Still Have Issues

1. **Network timeout on pip**: Just retry the pip install command
2. **Data not found**: Make sure you added the dataset to your notebook
3. **OOM error**: Reduce batch_size from 8 to 6 or 4
4. **Slow training**: Switch to T4 if P100 unavailable

Good luck! 🚀
