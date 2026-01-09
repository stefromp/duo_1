#!/bin/bash
# Quick Setup Script for Kaggle
# Run this after cloning the repo on Kaggle

echo "=== DUO Training Setup for Kaggle ==="
echo ""

# 1. Install dependencies
echo "Step 1: Installing dependencies..."
pip install -q -r requirements.txt
pip install -q flash-attn --no-build-isolation 2>/dev/null || echo "Flash attention skipped (optional)"

# 2. Create cache directory
echo "Step 2: Creating cache directory..."
mkdir -p cache

# 3. Check for integral cache
echo "Step 3: Checking integral cache..."
if [ ! -f "integral/gpt2.pkl" ]; then
    echo "Generating integral cache for GPT-2..."
    python utils.py --vocab_size=50257
else
    echo "Integral cache found!"
fi

# 4. Check dataset
echo "Step 4: Checking dataset..."
if [ -f "processed_data /train_subset_clean.txt" ]; then
    LINES=$(wc -l < "processed_data /train_subset_clean.txt")
    echo "Dataset found with $LINES lines"
else
    echo "WARNING: Dataset not found at 'processed_data /train_subset_clean.txt'"
    echo "Please upload your dataset file!"
fi

# 5. Test GPU
echo "Step 5: Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To start training, run:"
echo "  python main.py --config-name=config_kaggle"
echo ""
echo "For minimal resources, run:"
echo "  bash train_kaggle_minimal.sh"
echo ""
